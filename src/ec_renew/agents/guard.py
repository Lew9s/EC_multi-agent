"""守卫：run 级全局状态的**唯一写者**（design.md §5.4.2 / D-71；§12 1c 的逐条判据）。

元智能体的 act 是**请求**，不是执行。它是概率组件（今天接规则骨架，将来可接 LLM），
而 run 级状态里有两类东西是整个系统可信度的地基：

* ``EvidenceRegistry`` 的 ``_store`` 与 ``_baselines[round]`` —— **事实**；
* ``RunContext.projections``（``ProjectionRecord``）—— **审计**。

若元智能体能直接写它们，会同时坏掉三件事（§5.4.2）：造出的证据与检索到的证据无法
区分、不进事件日志就不可回放、同输入不同输出。因此「扩基线 / ``freeze()`` /
``register()`` / 写 ``ProjectionRecord`` / 扣额度」**只发生在本模块**。

判据的两条总原则（§12 1c 在此定稿）：

1. **越界 id 一律剔除并记 ``correct``，而不是整单作废** —— 元智能体是概率组件，
   一条坏 id 不该毁掉整轮。剔除后若已不满足下限（专家数 < 3、证据子集为空），
   才**整单作废**，由上层回落到规则骨架。
2. **未知载荷键直接作废**（``unknown_payload_key``）—— 这是让 D-75「分发通道零自由
   文本」成为**结构性**约束而不是承诺的那道闸门：提案里放不进任何自由文本字段。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ..contracts import (
    EXPERT_IDS,
    ActionProposal,
    ActivationPlan,
    AttributionProposal,
    EvidenceRequest,
    ExpertOpinion,
    ExpertTask,
    GuardVerdict,
    MemoryView,
    ProjectionRecord,
    RoundFold,
    SubQuestion,
    Usage,
)
from ..errors import InvalidRequest
from ..ports import EvidenceRegistryPort, RunContext
from .experts import MIN_SELECTED_EXPERTS
from .memory import MemoryService

#: 每轮的 ``read_memory`` 次数上限（防空转）。Q-20 尚未定值，此处取 3 作为占位，
#: 越界一律 ``reject``（不是 correct）——它没有可修正的形态，只是问得太频繁。
MAX_READS_PER_ROUND = 3

#: ``dispatch_experts`` 允许出现的载荷键。**闭集**：多一个键就作废（D-75）。
DISPATCH_PAYLOAD_KEYS = frozenset(
    {"active_experts", "weights", "evidence_scope", "cross_domain_flags"}
)

#: 每个动作允许的载荷键（同样是闭集）。
ACTION_PAYLOAD_KEYS: dict[str, frozenset[str]] = {
    "read_memory": frozenset({"view", "round"}),
    "dispatch_experts": DISPATCH_PAYLOAD_KEYS,
    "request_evidence": frozenset({"query", "reason", "scope", "effective_round"}),
    "attribute": frozenset({"round"}),
    "ask_human": frozenset({"reason"}),
    "finalize": frozenset(),
}

VALID_VIEWS = ("baseline", "opinions", "rounds")
VALID_SCOPES = ("components", "departments", "cases", "standards")


@dataclass
class LoopState:
    """循环的可变状态。**实例由 ``AgentRuntime`` 持有**，守卫只读它。

    放在这里而不是 runtime，是为了避免 ``guard ↔ runtime`` 的循环导入；守卫是
    「这份状态长什么样」的权威（它要按状态判定提案是否越界）。
    """

    round_no: int = 0
    active: tuple[str, ...] = ()
    weights: dict[str, float] = field(default_factory=dict)
    latest: dict[str, ExpertOpinion] = field(default_factory=dict)
    round_opinions: dict[int, dict[str, ExpertOpinion]] = field(default_factory=dict)
    consensus_score: float = 0.0
    baseline: list[str] = field(default_factory=list)
    plans: list[ActivationPlan] = field(default_factory=list)
    folds: list[RoundFold] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    evidence_requests: list[EvidenceRequest] = field(default_factory=list)
    attributions: list[AttributionProposal] = field(default_factory=list)
    last_view: MemoryView | None = None


class Guard:
    """裁决提案 + 执行特权写入。**没有任何其它模块可以改这些状态。**"""

    def __init__(self, ctx: RunContext, *, memory: MemoryService) -> None:
        self._ctx = ctx
        self._memory = memory
        self._reads: dict[int, int] = defaultdict(int)
        self._human_asked = 0
        self._requests: list[EvidenceRequest] = []
        self._attributions: list[AttributionProposal] = []

    # -- 只读观测（供上层与测试） ------------------------------------------- #
    @property
    def evidence_requests(self) -> list[EvidenceRequest]:
        return list(self._requests)

    @property
    def attributions(self) -> list[AttributionProposal]:
        return list(self._attributions)

    # -- 裁决 --------------------------------------------------------------- #
    def judge(self, proposal: ActionProposal, state: LoopState) -> GuardVerdict:
        """逐条判据。返回 ``accept`` / ``correct`` / ``reject`` 与原因。"""
        allowed = ACTION_PAYLOAD_KEYS.get(proposal.action)
        if allowed is None:
            return GuardVerdict(action="reject", reason=f"unknown_action:{proposal.action}")
        unknown = sorted(set(proposal.payload) - allowed)
        if unknown:
            # 让 D-75 成为结构性约束的那道闸门：放不进自由文本字段。
            return GuardVerdict(
                action="reject",
                reason="unknown_payload_key",
                corrections=[f"非法载荷键：{unknown}"],
            )
        return getattr(self, f"_judge_{proposal.action}")(proposal.payload, state)

    def _judge_read_memory(self, payload: dict, state: LoopState) -> GuardVerdict:
        view = str(payload.get("view", ""))
        if view not in VALID_VIEWS:
            return GuardVerdict(action="reject", reason=f"unknown_view:{view}")
        if self._reads[state.round_no] >= MAX_READS_PER_ROUND:
            return GuardVerdict(action="reject", reason="read_memory_budget_exhausted")
        return GuardVerdict(action="accept")

    def _judge_dispatch_experts(self, payload: dict, state: LoopState) -> GuardVerdict:
        plan, corrections = self._sanitize_plan(payload, state)
        if plan is None:
            return GuardVerdict(action="reject", reason="too_few_experts", corrections=corrections)
        if corrections:
            return GuardVerdict(action="correct", corrections=corrections)
        return GuardVerdict(action="accept")

    def _judge_request_evidence(self, payload: dict, state: LoopState) -> GuardVerdict:
        if not str(payload.get("query", "")).strip() or not str(payload.get("reason", "")).strip():
            return GuardVerdict(action="reject", reason="evidence_request_incomplete")
        if str(payload.get("scope", "")) not in VALID_SCOPES:
            return GuardVerdict(action="reject", reason="unknown_scope")
        corrections: list[str] = []
        # D-16：请求只能对**下一轮**生效，不得中途改本轮已冻结的基线。
        if int(payload.get("effective_round", 0) or 0) <= state.round_no:
            corrections.append("effective_round 已改为下一轮（D-16）")
        return GuardVerdict(action="correct" if corrections else "accept", corrections=corrections)

    def _judge_attribute(self, payload: dict, state: LoopState) -> GuardVerdict:
        # 第 1 轮没有「上一轮分歧」可归因。
        if state.round_no < 2:
            return GuardVerdict(action="reject", reason="attribute_before_second_round")
        return GuardVerdict(action="accept")

    def _judge_ask_human(self, payload: dict, state: LoopState) -> GuardVerdict:
        if self._human_asked >= 1:
            return GuardVerdict(action="reject", reason="ask_human_already_used")
        return GuardVerdict(action="accept")

    def _judge_finalize(self, payload: dict, state: LoopState) -> GuardVerdict:
        if not state.latest:
            return GuardVerdict(action="reject", reason="no_opinions")
        known = set(self._ctx.registry.all_ids())
        for expert, opinion in sorted(state.latest.items()):
            cited = set(opinion.evidence_ids)
            if not cited or not cited <= known:
                # §5.4.1：任一意见无支撑就拒绝收束，转 manual_review。
                return GuardVerdict(
                    action="reject", reason="unsupported_opinion", corrections=[expert]
                )
        return GuardVerdict(action="accept")

    # -- 特权写入 ----------------------------------------------------------- #
    def _sanitize_plan(
        self, payload: Mapping[str, object], state: LoopState
    ) -> tuple[ActivationPlan | None, list[str]]:
        """把提案里的 ``ActivationPlan`` 收敛到合法形态。

        返回 ``(plan, corrections)``；``plan is None`` 表示整单作废。
        """
        corrections: list[str] = []
        raw_active = payload.get("active_experts") or []
        active = [str(e) for e in raw_active if str(e) in EXPERT_IDS]  # type: ignore[union-attr]
        unknown = sorted({str(e) for e in raw_active} - set(EXPERT_IDS))  # type: ignore[union-attr]
        if unknown:
            corrections.append(f"剔除未知专家 id：{unknown}")

        # D-19：专家集自动修订**只增不减**；已激活的不许被移除。
        for expert in state.active:
            if expert not in active:
                active.append(expert)
                corrections.append(f"不得移除已激活专家，已加回：{expert}")

        ordered = [e for e in EXPERT_IDS if e in set(active)]
        if len(ordered) < MIN_SELECTED_EXPERTS:
            return None, corrections + [f"专家数不足 {MIN_SELECTED_EXPERTS}"]

        raw_weights = payload.get("weights") or {}
        weights: dict[str, float] = {}
        if isinstance(raw_weights, Mapping):
            for expert, value in raw_weights.items():
                key = str(expert)
                if key in ordered:
                    weights[key] = max(0.0, float(value))  # type: ignore[arg-type]
        dropped = sorted(set(weights) ^ set(ordered))
        if dropped:
            weights = {e: weights.get(e, 0.0) for e in ordered}
        if not any(weights.values()):
            weights = {e: 1.0 for e in ordered}
            corrections.append("权重全为 0，已回落到均匀权重")
        total = sum(weights.values())
        # 归一化到 Σ=1（§5.4.7）。**不做四舍五入**：取整后再与未取整的原值比较，
        # 会让「骨架已经给了 Σ=1 的权重」也每轮误报一次 correct。
        normalized = {e: w / total for e, w in weights.items()}
        if any(abs(normalized[e] - weights[e]) > 1e-9 for e in weights):
            corrections.append("权重已归一化到 Σ=1")
        weights = normalized

        known = set(self._ctx.registry.all_ids())
        baseline = self._ctx.registry.baseline(state.round_no or 1) or sorted(known)
        raw_scope = payload.get("evidence_scope") or {}
        scope: dict[str, list[str]] = {}
        if isinstance(raw_scope, Mapping):
            for expert, ids in raw_scope.items():
                key = str(expert)
                if key not in ordered:
                    continue
                kept = sorted({str(i) for i in ids if str(i) in known})  # type: ignore[union-attr]
                if len(kept) != len(list(ids)):  # type: ignore[arg-type]
                    corrections.append(f"{key} 的证据子集剔除了越界 id（D-25）")
                if not kept:
                    # 无证据 → 不给该专家派发事实（§4 不变量 4），回落本轮基线。
                    kept = list(baseline)
                    corrections.append(f"{key} 的证据子集为空，回落本轮基线")
                scope[key] = kept
        for expert in ordered:
            scope.setdefault(expert, list(baseline))

        flags = [str(f) for f in (payload.get("cross_domain_flags") or []) if str(f) in EXPERT_IDS]  # type: ignore[union-attr]
        return (
            ActivationPlan(
                active_experts=ordered,
                weights=weights,
                evidence_scope=scope,
                rationale="",
                cross_domain_flags=flags,
            ),
            corrections,
        )

    def dispatch(
        self,
        payload: Mapping[str, object],
        *,
        state: LoopState,
        request: str,
        sub_questions: Sequence[SubQuestion] = (),
    ) -> list[ExpertTask]:
        """冻结本轮基线 + 确定性投影。**只有这里会写 registry / projections。**"""
        plan, corrections = self._sanitize_plan(payload, state)
        if plan is None:
            raise InvalidRequest("守卫已作废该派发提案：" + "；".join(corrections))

        round_no = state.round_no
        # 每轮重算基线：run 中途登记的证据（人类补充的事实、将来的 EvidenceRequest）
        # 属于下一轮冻结快照。
        baseline = self._ctx.registry.all_ids()
        self._ctx.registry.freeze(round_no, baseline)
        state.baseline = list(baseline)
        self._ctx.events.emit("round_started", round=round_no, experts=plan.active_experts)

        tasks: list[ExpertTask] = []
        dissent = sum(1 for op in state.latest.values() if op.decision != "approve")
        for expert in plan.active_experts:
            task, record = self._memory.project(
                expert=expert,
                request=request,
                round_no=round_no,
                sub_questions=list(sub_questions),
                previous=state.latest.get(expert),
                opinions=state.latest.values(),
                consensus_score=state.consensus_score,
                dissent_count=dissent,
            )
            # evidence_scope 是元智能体唯一能影响「给谁看什么」的地方，因此必须
            # 用它来裁剪投影，而不是原样给全量基线（D-75 的可见性由守卫落实）。
            allowed = set(plan.evidence_scope.get(expert, baseline))
            task = task.model_copy(
                update={"evidence": [e for e in task.evidence if e.evidence_id in allowed]}
            )
            tasks.append(task)
            self._ctx.projections.append(record)
            self._ctx.events.emit(
                "projected",
                round=round_no,
                expert=expert,
                policy=record.policy.value,
                disclosed=len(record.disclosed_claim_ids),
            )

        state.active = tuple(plan.active_experts)
        state.weights = dict(plan.weights)
        state.plans.append(plan)
        self._ctx.events.emit(
            "experts_selected", active=list(state.active), weights=dict(state.weights)
        )
        return tasks

    def resident(self, *, state: LoopState) -> MemoryView:
        """常驻层（§5.4.3 的 L2/L3 聚合部分）：证据 id 全集 + 计数 + 本轮决策。

        **不含 L4 的轮次折叠**——那一层按需经 ``read_memory(view="rounds")`` 取
        （D-74：L4 不常驻）。本方法是纯读，不占 ``read_memory`` 的额度。
        """
        baseline = self._ctx.registry.baseline(state.round_no) or self._ctx.registry.all_ids()
        return MemoryView(
            view="baseline",
            round=state.round_no,
            evidence_ids=list(baseline),
            counts={
                "registry": len(self._ctx.registry.all_ids()),
                "opinions": len(state.latest),
                "graph": self._ctx.registry.count_by_source("graph"),
            },
            decisions={
                expert: opinion.decision for expert, opinion in sorted(state.latest.items())
            },
        )

    def read(self, view: str, *, state: LoopState) -> MemoryView:
        """读全局记忆的某个枚举视图；这是元智能体唯一被授权的读入口（D-53）。"""
        if view not in VALID_VIEWS:
            raise InvalidRequest(f"未知视图：{view!r}")
        self._reads[state.round_no] += 1
        return self._memory.read_view(
            view, round_no=state.round_no, opinions=state.latest.values(), folds=state.folds
        )

    def record_evidence_request(self, payload: Mapping[str, object], *, state: LoopState) -> EvidenceRequest:
        """登记证据请求。**只登记，不改本轮冻结基线**（D-16）。"""
        request = EvidenceRequest(
            query=str(payload.get("query", "")),
            reason=str(payload.get("reason", "")),
            scope=str(payload.get("scope", "cases")),  # type: ignore[arg-type]
            effective_round=max(state.round_no + 1, int(payload.get("effective_round", 0) or 0)),
        )
        self._requests.append(request)
        self._ctx.events.emit(
            "evidence_requested",
            round=state.round_no,
            effective_round=request.effective_round,
            scope=request.scope,
        )
        return request

    def attribute(self, *, state: LoopState) -> AttributionProposal:
        """分歧归因：**确定性**算证据重叠，产出提案（Q-22：先记录、不驱动补救）。"""
        opinion = self._attribute_deterministically(state.latest)
        self._attributions.append(opinion)
        self._ctx.events.emit(
            "attributed",
            round=state.round_no,
            kind=opinion.kind,
            evidence_overlap=opinion.evidence_overlap,
            divergent=opinion.divergent_evidence,
        )
        return opinion

    @staticmethod
    def _attribute_deterministically(opinions: Mapping[str, ExpertOpinion]) -> AttributionProposal:
        """按 §5.2.6：先算证据集 Jaccard，再据它分类。

        分类规则**确定性、可复现**，且不驱动任何补救路径（Q-22：阈值未标定）：

        * 找不到「判断不同」的专家对 → 无可归因（``judgment``，重叠 1.0）；
        * 重叠 == 1（证据相同）→ ``judgment``：换多少证据都改不了，是判断分歧；
        * 重叠 == 0（证据完全不同）→ ``evidence``：分歧与证据切分完全重合；
        * ``0 < 重叠 < 1`` → ``mixed``：这正是 Q-22 所说的边界区间，在标定前只记录。
        """
        ranked = sorted(opinions.items())
        if len(ranked) < 2:
            return AttributionProposal(round=0, kind="judgment", evidence_overlap=1.0)
        head = ranked[0]
        partner = next(
            (pair for pair in ranked[1:] if pair[1].decision != head[1].decision), None
        )
        if partner is None:
            return AttributionProposal(round=0, kind="judgment", evidence_overlap=1.0)

        left, right = set(head[1].evidence_ids), set(partner[1].evidence_ids)
        union = left | right
        overlap = len(left & right) / len(union) if union else 1.0
        divergent = sorted(union - (left & right))
        if overlap >= 1.0:
            kind = "judgment"
        elif overlap <= 0.0:
            kind = "evidence"
        else:
            kind = "mixed"
        return AttributionProposal(
            round=0,
            kind=kind,  # type: ignore[arg-type]
            evidence_overlap=round(overlap, 4),
            divergent_evidence=divergent,
        )

    def ask_human(self, payload: Mapping[str, object], *, state: LoopState) -> None:
        """登记后置 HITL 待办。**决定何时真正挂起的是外环**（D-73）。"""
        self._human_asked += 1
        self._ctx.events.emit(
            "pending_human_review",
            round=state.round_no,
            reason=str(payload.get("reason", "consensus_not_reached")),
        )

    def evidence_ids(self) -> list[str]:
        return sorted(self._ctx.registry.all_ids())

    def register_human_facts(self, facts: Iterable[str]) -> None:
        """人类提供的事实登记为 ``source="human"`` 的**一等证据**（D-28）。

        这是守卫的另一处特权写入：``register()`` 不在业务模块里出现。
        """
        for raw in facts:
            self._ctx.registry.register(source="human", content=raw)


def projection_records(ctx: RunContext) -> list[ProjectionRecord]:
    """审计视图（只读）：向谁披露了什么。"""
    return list(ctx.projections)


def known_evidence_ids(registry: EvidenceRegistryPort) -> list[str]:
    return sorted(registry.all_ids())
