"""harness：元智能体与子智能体**共用的唯一执行入口**（design.md §3.1 / D-84 的竖切）。

三条硬约束在这里落地：

* **唯一执行路径**：元智能体 = 一个 ``AgentSpec`` 跑 think-act-observe 循环；子智能体 =
  同一个 runtime 的另一次执行。两者差别只在 ``AgentSpec`` 的五项数据，不存在第二条路径。
* **act 是请求**：runtime 从不直接写 registry / projections / 额度——一律先交
  ``Guard.judge()`` 裁决，再由守卫执行特权写入（D-71 / §5.4.2）。
* **逐 step 可回放**：每次 think / act / guard / llm 都在**执行前**落一条 ``step``，
  执行后落一条 ``step_done``（§9.4 / D-85）。append-only 的 JSONL 改不了前一条，
  所以用成对事件——前一条正是「进程崩了也不重复付费」的凭据。

裁定权（共识判定、停滞判定）**由外环以回调注入**（D-73：顺序权归内环，额度权与裁定权
归外环）。这样内环不必 import `workflow`，单向依赖在目录层面就成立。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ..contracts import (
    META_ACTIONS,
    ActionProposal,
    AgentKind,
    ExpertOpinion,
    ExpertTask,
    GuardVerdict,
    MemoryView,
    RoundFold,
    StallReport,
    StepRecord,
    SubQuestion,
    Usage,
)
from ..errors import BudgetExceeded, InvariantViolation, StepLimitExceeded
from ..ports import RunContext
from .guard import Guard, LoopState
from .memory import MemoryService
from .skills.expert_review import abstain_opinion, run_expert


@dataclass(frozen=True)
class AgentSpec:
    """一个 agent 的全部差异（§3.1）：输入/输出 schema、校验器、工具集、动作空间。

    深度**数值化**（D-79）：元智能体 0、子智能体 1。深度 ≥ 1 时 ``dispatch_experts``
    不可见（``visible_acts``），这比「禁止再次委派」那句措辞可校验。
    """

    name: str
    kind: AgentKind
    depth: int = 0
    acts: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()

    @property
    def visible_acts(self) -> tuple[str, ...]:
        if self.depth >= 1:
            return tuple(a for a in self.acts if a != "dispatch_experts")
        return self.acts


#: 子智能体的 spec：深度 1、动作空间为空、看不到 ``dispatch_experts``（D-78 / D-79）。
EXPERT_SPEC = AgentSpec(
    name="expert", kind="expert", depth=1, acts=(), tools=("search_evidence",)
)

#: 元智能体可见的工具集。``read_memory`` 同时出现在 act 与 tool 里，因为它是元智能体
#: 唯一被授权的**读**工具（§5.4.11 / D-53）；`dispatch_experts` 只在 act 里，不在工具集。
META_SPEC = AgentSpec(
    name="meta", kind="meta", depth=0, acts=META_ACTIONS, tools=("read_memory",)
)


@dataclass
class MetaLoopResult:
    """一轮 run 的循环产出。外环据此做后段（渲染 / 保证等级 / 振荡记录）。"""

    opinions_by_round: dict[int, dict[str, ExpertOpinion]] = field(default_factory=dict)
    latest: dict[str, ExpertOpinion] = field(default_factory=dict)
    score: float = 0.0
    effective: int = 0
    status: str = "retry"
    stall: StallReport = field(default_factory=StallReport)
    usage: Usage = field(default_factory=Usage)
    warnings: list[str] = field(default_factory=list)
    rounds: int = 0
    active: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    plans: list[object] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)


class AgentRuntime:
    """think-act-observe 的唯一执行入口。元与子智能体**共用这一个类**。"""

    def __init__(
        self,
        ctx: RunContext,
        *,
        guard: Guard,
        memory: MemoryService,
        run_id: str = "",
    ) -> None:
        self._ctx = ctx
        self._guard = guard
        self._memory = memory
        self._run_id = run_id or getattr(ctx, "run_id", "")
        self._steps: list[StepRecord] = []
        self._counters: dict[tuple[str, int], int] = {}

    # ------------------------------------------------------------------ #
    # StepRecord
    # ------------------------------------------------------------------ #
    def _step(
        self, *, node: str, round_no: int, kind: str, args_hash: str = ""
    ) -> StepRecord:
        """**执行前**落盘（§9.4）：崩在这里也不会重复付费。"""
        key = (node, round_no)
        index = self._counters.get(key, 0)
        self._counters[key] = index + 1
        record = StepRecord(
            run_id=self._run_id,
            node=node,
            round=round_no,
            step=index,
            kind=kind,  # type: ignore[arg-type]
            args_hash=args_hash,
        )
        self._steps.append(record)
        self._ctx.events.emit("step", **record.model_dump())
        return record

    def _done(self, record: StepRecord, *, produced: Sequence[str] = (), outcome: str = "ok") -> None:
        self._ctx.events.emit(
            "step_done",
            run_id=record.run_id,
            node=record.node,
            round=record.round,
            step=record.step,
            kind=record.kind,
            produced_ids=[str(p) for p in produced],
            outcome=outcome,
        )

    @staticmethod
    def _args_hash(payload: Mapping[str, object]) -> str:
        """参数指纹。**不含自由文本原文**——只留摘要，日志里读不出提案内容。"""
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    @property
    def steps(self) -> list[StepRecord]:
        return list(self._steps)

    # ------------------------------------------------------------------ #
    # act：先裁决，再由守卫执行特权写入
    # ------------------------------------------------------------------ #
    async def act(
        self,
        proposal: ActionProposal,
        state: LoopState,
        *,
        request: str,
        sub_questions: Sequence[SubQuestion] = (),
    ) -> GuardVerdict:
        node = "meta" if META_SPEC.depth == 0 else "agent"
        act_step = self._step(
            node=node,
            round_no=state.round_no,
            kind="act",
            args_hash=self._args_hash({"action": proposal.action, **proposal.payload}),
        )
        guard_step = self._step(node="guard", round_no=state.round_no, kind="guard")
        verdict = self._guard.judge(proposal, state)
        self._ctx.events.emit(
            "guard_verdict",
            round=state.round_no,
            action=proposal.action,
            verdict=verdict.action,
            reason=verdict.reason,
            corrections=list(verdict.corrections),
        )
        if verdict.action == "reject":
            self._done(guard_step, outcome="rejected")
            self._done(act_step, outcome="rejected")
            return verdict

        produced: list[str] = []
        if proposal.action == "read_memory":
            view = self._guard.read(str(proposal.payload.get("view", "baseline")), state=state)
            state.last_view = view
            produced = list(view.evidence_ids[:8])
        elif proposal.action == "dispatch_experts":
            tasks = self._guard.dispatch(
                proposal.payload, state=state, request=request, sub_questions=list(sub_questions)
            )
            opinions, spent = await self.dispatch_experts(tasks)
            state.round_opinions[state.round_no] = dict(opinions)
            state.latest = dict(opinions)
            state.usage = state.usage + spent
            produced = sorted(opinions)
        elif proposal.action == "request_evidence":
            state.evidence_requests.append(
                self._guard.record_evidence_request(proposal.payload, state=state)
            )
        elif proposal.action == "attribute":
            state.attributions.append(self._guard.attribute(state=state))
        elif proposal.action == "ask_human":
            self._guard.ask_human(proposal.payload, state=state)
        elif proposal.action == "finalize":
            pass  # 收束本身没有副作用；渲染与保证等级由外环的后段负责
        else:  # pragma: no cover - 动作空间是闭集，走到这里说明枚举与实现脱节
            raise InvariantViolation(f"动作空间里没有 {proposal.action!r} 的实现")

        self._done(guard_step)
        self._done(act_step, produced=produced)
        return verdict

    # ------------------------------------------------------------------ #
    # 子智能体：同一 runtime 的另一次执行（唯一执行入口）
    # ------------------------------------------------------------------ #
    async def _run_one(
        self, task: ExpertTask, allowed: Sequence[str]
    ) -> tuple[ExpertOpinion, Usage]:
        """Expected failures become values; fatal errors stay exceptions.

        与重构前逐字等价：超时/传输错误在这里变成 ``abstain`` **值**，TaskGroup 因此
        看不到它，兄弟专家得以跑完（§7.6）。
        """
        step = self._step(node=f"expert:{task.expert}", round_no=task.round, kind="llm")
        try:
            async with asyncio.timeout(task.budget.timeout_s):
                opinion, usage = await run_expert(task, self._ctx.llm, allowed, events=self._ctx.events)
            self._done(step, produced=list(opinion.evidence_ids))
            return opinion, usage
        except asyncio.CancelledError:
            raise
        except (InvariantViolation, BudgetExceeded, StepLimitExceeded):
            raise
        except Exception as exc:  # noqa: BLE001 — 有意的「异常转状态」点，见下方说明
            self._ctx.events.emit(
                "failure",
                node="expert",
                expert=task.expert,
                cause_type=type(exc).__name__,
                message=str(exc)[:200],
            )
            self._done(step, outcome="failed")
            return abstain_opinion(task.expert, allowed, f"{type(exc).__name__}"), Usage(calls=1)

    async def dispatch_experts(
        self, tasks: Sequence[ExpertTask]
    ) -> tuple[dict[str, ExpertOpinion], Usage]:
        allowed = self._ctx.registry.all_ids()
        async with asyncio.TaskGroup() as group:
            running = [group.create_task(self._run_one(task, allowed)) for task in tasks]

        opinions: dict[str, ExpertOpinion] = {}
        usage = Usage()
        # Sorted merge -> order independent, identical on replay.
        for task, runner in sorted(zip(tasks, running), key=lambda pair: pair[0].expert):
            opinion, spent = runner.result()
            opinions[task.expert] = opinion
            usage = usage + spent
        return opinions, usage

    # ------------------------------------------------------------------ #
    # 元智能体主循环
    # ------------------------------------------------------------------ #
    async def run_meta(
        self,
        spec: AgentSpec,
        *,
        think: Callable[[LoopState, MemoryView], list[ActionProposal]],
        closing: Callable[[bool, bool], ActionProposal],
        request: str,
        sub_questions: Sequence[SubQuestion] = (),
        max_rounds: int = 3,
        consensus_fn: Callable[
            [dict[str, ExpertOpinion], dict[str, float]], tuple[float, int, str]
        ],
        stall_fn: Callable[
            [dict[str, ExpertOpinion], dict[str, ExpertOpinion], Sequence[str], Sequence[str]], bool
        ],
    ) -> MetaLoopResult:
        """一个评审轮 = 一次 think-act-observe；``max_rounds`` 就是轮次上限本身，
        不引入第二套「元智能体步数上限」（§5.4.1）。"""
        if spec.kind != "meta" or spec.depth != 0:
            raise InvariantViolation("run_meta 只能跑元智能体的 spec（depth=0）")

        state = LoopState()
        result = MetaLoopResult()
        previous: dict[str, ExpertOpinion] = {}
        previous_baseline: list[str] = []

        for round_no in range(1, max_rounds + 1):
            state.round_no = round_no
            result.rounds = round_no

            # --- think（§5.4.1：读骨架 + 记忆视图 → 定激活集与证据分配） ---
            think_step = self._step(
                node="meta", round_no=round_no, kind="think", args_hash=self._args_hash({"round": round_no})
            )
            resident = self._guard.resident(state=state)
            proposals = think(state, resident)
            self._done(think_step, produced=[p.action for p in proposals])

            for proposal in proposals:
                if proposal.action != "dispatch_experts":
                    await self.act(
                        proposal, state, request=request, sub_questions=sub_questions
                    )

            dispatch = next((p for p in proposals if p.action == "dispatch_experts"), None)
            if dispatch is None:  # pragma: no cover - 骨架契约：每轮必须有一次派发
                raise InvariantViolation("元智能体在这一轮没有给出 dispatch_experts")
            verdict = await self.act(
                dispatch, state, request=request, sub_questions=sub_questions
            )
            if verdict.action == "reject":
                # 守卫整单作废（§12 1c）：本轮没有意见可评，显式交人工而不是静默继续。
                result.warnings.append(f"dispatch_rejected:{verdict.reason}")
                result.status = "manual_review"
                break

            # --- observe（裁定权在外环：回调注入） ------------------------ #
            score, effective, status = consensus_fn(state.latest, state.weights)
            state.consensus_score = score
            self._ctx.events.emit(
                "round_finished",
                round=round_no,
                consensus_score=score,
                effective_experts=effective,
                status=status,
                decisions={e: op.decision for e, op in sorted(state.latest.items())},
            )
            state.folds.append(
                RoundFold(
                    round=round_no,
                    consensus_score=score,
                    dissent_count=sum(
                        1 for op in state.latest.values() if op.decision != "approve"
                    ),
                    active_experts=list(state.active),
                )
            )
            if status != "retry":
                break

            # --- 不动点（D-81）：只从第 2 轮起有意义 -------------------- #
            if previous and stall_fn(previous, state.latest, previous_baseline, state.baseline):
                result.stall = StallReport(
                    stalled=True,
                    detected_at_round=round_no,
                    skipped_rounds=max_rounds - round_no,
                    unchanged_experts=sorted(state.latest),
                )
                self._ctx.events.emit(
                    "stalled", round=round_no, experts=result.stall.unchanged_experts
                )
                self._ctx.events.emit("stall_skipped", skipped_rounds=result.stall.skipped_rounds)
                # 不是 manual_review：没有专家仍在分歧，只是流程不再产生信息（§5.6.1）。
                status = "stalled"
                result.warnings.append("stalled")
                break

            previous = dict(state.latest)
            previous_baseline = list(state.baseline)

        if status == "retry":
            status = "manual_review"
            result.warnings.append("max_rounds_reached")

        # --- 收束 act：达标/停滞 → finalize，未达共识 → ask_human（§5.4.1） --- #
        state.round_no = result.rounds
        await self.act(
            closing(status == "approved", status == "stalled"),
            state,
            request=request,
            sub_questions=sub_questions,
        )

        result.opinions_by_round = dict(state.round_opinions)
        result.latest = dict(state.latest)
        result.score = state.consensus_score
        result.effective = sum(
            1 for op in state.latest.values() if op.decision != "abstain"
        )
        result.status = status
        result.usage = state.usage
        result.active = list(state.active)
        result.weights = dict(state.weights)
        result.plans = list(state.plans)
        result.steps = self.steps
        return result
