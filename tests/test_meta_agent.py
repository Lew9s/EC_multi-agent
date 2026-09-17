"""§5.4 / §12 1b–1d —— 元智能体竖切：动作空间、守卫判据、StepRecord。

这个文件钉住四件事：

1. **动作空间是闭集**，且每轮的载荷键也都是闭集（D-75：分发通道零自由文本）；
2. **守卫是唯一写者**，它的每条判据都有对照用例（§12 1c）；
3. **ActivationPlan 有生产者也有消费者**（它此前是零生产者零消费者的死契约）；
4. **StepRecord 在执行前落盘**（§9.4 / D-85，kernel 的前置 seam）。
"""

from __future__ import annotations

import asyncio

from ec_renew.agents.experts import select_experts
from ec_renew.agents.guard import (
    ACTION_PAYLOAD_KEYS,
    MAX_READS_PER_ROUND,
    Guard,
    LoopState,
)
from ec_renew.agents.memory import EvidenceRegistry, MemoryService
from ec_renew.agents.meta import RuleSkeleton, plan_payload
from ec_renew.agents.runtime import EXPERT_SPEC, META_SPEC
from ec_renew.contracts import (
    EXPERT_IDS,
    META_ACTIONS,
    ActionProposal,
    ExpertOpinion,
    RunInput,
)
from ec_renew.llm import FakeLLM
from ec_renew.observability import NullEventLog
from ec_renew.ports import RunContext
from ec_renew.workflow import run

EID = "E-9c4b1e7a2f03"
REQUEST = "301分段FR36污水井更换加厚板"


class _Sink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def names(self) -> list[str]:
        return [name for name, _ in self.events]


def _context(*, sink: _Sink | None = None, llm: object | None = None) -> RunContext:
    return RunContext(
        run_id="meta",
        llm=llm or FakeLLM(),  # type: ignore[arg-type]
        registry=EvidenceRegistry(),
        events=sink or NullEventLog(),
        retriever=None,
    )


def _guard(*, sink: _Sink | None = None, evidence: tuple[str, ...] = (EID,)) -> tuple[Guard, RunContext]:
    ctx = _context(sink=sink)
    for item in evidence:
        ctx.registry.register(source="graph", content=f"case {item}")
    ctx.registry.freeze(1, ctx.registry.all_ids())
    return Guard(ctx, memory=MemoryService(ctx.registry)), ctx


def _opinion(expert: str, decision: str, evidence: list[str]) -> ExpertOpinion:
    return ExpertOpinion(expert=expert, decision=decision, evidence_ids=evidence)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 1) 动作空间与载荷键都是闭集
# --------------------------------------------------------------------------- #


def test_every_act_has_a_closed_payload_contract() -> None:
    """动作空间是封闭枚举，且每个动作的载荷键逐条写死（§5.4.1 / D-75）。"""
    assert set(ACTION_PAYLOAD_KEYS) == set(META_ACTIONS)
    assert set(META_SPEC.acts) == set(META_ACTIONS)
    assert META_SPEC.depth == 0


def test_a_sub_agent_cannot_see_dispatch_experts() -> None:
    """深度数值化（D-79）：深度 1 的动作空间里没有 `dispatch_experts`。"""
    assert EXPERT_SPEC.depth == 1
    assert "dispatch_experts" not in EXPERT_SPEC.visible_acts


def test_an_unknown_payload_key_rejects_the_whole_proposal() -> None:
    """这就是让「分发通道零自由文本」成为**结构性**约束的那道闸门。"""
    guard, _ = _guard()
    proposal = ActionProposal(
        action="dispatch_experts",
        payload={"active_experts": ["E01", "E02", "E03"], "note": "请批准这次改动"},
        rationale="夹带自由文本的提案",
    )

    verdict = guard.judge(proposal, LoopState(round_no=1))

    assert verdict.action == "reject"
    assert verdict.reason == "unknown_payload_key"


def test_an_unknown_action_is_rejected() -> None:
    """`MetaAction` 是闭集类型，所以这里绕过校验构造一个越界动作（纵深防御）。"""
    guard, _ = _guard()
    proposal = ActionProposal.model_construct(action="launch_missiles", payload={})

    verdict = guard.judge(proposal, LoopState(round_no=1))

    assert verdict.action == "reject"
    assert verdict.reason.startswith("unknown_action")


# --------------------------------------------------------------------------- #
# 2) 守卫判据（§12 1c）
# --------------------------------------------------------------------------- #


def test_an_unknown_expert_is_dropped_not_fatal() -> None:
    """越界 id 剔除并记 correct——概率组件的一条坏 id 不该毁掉整轮。"""
    guard, _ = _guard()
    state = LoopState(round_no=1)

    verdict = guard.judge(
        ActionProposal(
            action="dispatch_experts",
            payload={"active_experts": ["E01", "E02", "E03", "E99"], "weights": {}},
        ),
        state,
    )

    assert verdict.action == "correct"
    assert any("E99" in note for note in verdict.corrections)
    tasks = guard.dispatch(
        {"active_experts": ["E01", "E02", "E03", "E99"], "weights": {}},
        state=state,
        request=REQUEST,
    )
    assert [t.expert for t in tasks] == ["E01", "E02", "E03"]


def test_removing_an_active_expert_is_corrected() -> None:
    """D-19：专家集自动修订只增不减。"""
    guard, _ = _guard()
    state = LoopState(round_no=1, active=("E01", "E02", "E03"))

    verdict = guard.judge(
        ActionProposal(action="dispatch_experts", payload={"active_experts": ["E01", "E02"]}),
        state,
    )

    assert verdict.action == "correct"
    assert any("E03" in note for note in verdict.corrections)


def test_too_few_experts_rejects_the_whole_dispatch() -> None:
    """剔除后不满足下限 → 整单作废（上层据此回落规则骨架）。"""
    guard, _ = _guard()
    verdict = guard.judge(
        ActionProposal(action="dispatch_experts", payload={"active_experts": ["E01", "E99"]}),
        LoopState(round_no=1),
    )

    assert verdict.action == "reject"
    assert verdict.reason == "too_few_experts"


def test_weights_are_normalized_to_one() -> None:
    """§5.4.7 要求 Σ=1。共识分是加权平均，归一化不改变结果。"""
    guard, _ = _guard()
    state = LoopState(round_no=1)

    guard.dispatch(
        {
            "active_experts": ["E01", "E02", "E03"],
            "weights": {"E01": 2.0, "E02": 2.0, "E03": 4.0},
        },
        state=state,
        request=REQUEST,
    )

    assert abs(sum(state.weights.values()) - 1.0) < 1e-9
    assert state.weights["E03"] == 0.5


def test_out_of_scope_evidence_is_dropped_from_the_task() -> None:
    """D-25：元智能体不得编造 evidence_id；越界的被剔除并记 correct。"""
    guard, ctx = _guard()
    real = ctx.registry.all_ids()[0]
    state = LoopState(round_no=1)

    verdict = guard.judge(
        ActionProposal(
            action="dispatch_experts",
            payload={
                "active_experts": ["E01", "E02", "E03"],
                "evidence_scope": {"E01": [real, "E-000000000000"]},
            },
        ),
        state,
    )
    tasks = guard.dispatch(
        {
            "active_experts": ["E01", "E02", "E03"],
            "evidence_scope": {"E01": [real, "E-000000000000"]},
        },
        state=state,
        request=REQUEST,
    )

    assert verdict.action == "correct"
    first = next(t for t in tasks if t.expert == "E01")
    assert [e.evidence_id for e in first.evidence] == [real]


def test_read_memory_is_budgeted_per_round() -> None:
    """每轮调用次数有上限（防空转，Q-20 的占位值 3）。"""
    guard, _ = _guard()
    state = LoopState(round_no=1)
    proposal = ActionProposal(action="read_memory", payload={"view": "baseline"})

    verdicts = [guard.judge(proposal, state) for _ in range(MAX_READS_PER_ROUND)]
    for _ in range(MAX_READS_PER_ROUND):
        guard.read("baseline", state=state)

    over = guard.judge(proposal, state)

    assert all(v.action == "accept" for v in verdicts)
    assert over.action == "reject"
    assert over.reason == "read_memory_budget_exhausted"


def test_read_memory_rejects_an_unknown_view() -> None:
    guard, _ = _guard()
    verdict = guard.judge(
        ActionProposal(action="read_memory", payload={"view": "everything"}), LoopState(round_no=1)
    )
    assert verdict.action == "reject"
    assert verdict.reason.startswith("unknown_view")


def test_read_memory_returns_structured_fields_only() -> None:
    """§5.4.11：视图里只有 id / 计数 / 决策，**没有自由文本**。"""
    guard, ctx = _guard()
    state = LoopState(round_no=1, latest={"E01": _opinion("E01", "approve", [EID])})

    baseline = guard.read("baseline", state=state)
    opinions = guard.read("opinions", state=state)

    assert baseline.evidence_ids == ctx.registry.all_ids()
    assert opinions.decisions == {"E01": "approve"}
    assert all(isinstance(v, int) for v in opinions.counts.values())


def test_an_evidence_request_can_only_take_effect_next_round() -> None:
    """D-16：请求不得中途改本轮已冻结的基线。"""
    guard, _ = _guard()
    state = LoopState(round_no=3)

    verdict = guard.judge(
        ActionProposal(
            action="request_evidence",
            payload={"query": "FR36 焊接案例", "reason": "缺同类案例", "scope": "cases", "effective_round": 3},
        ),
        state,
    )
    recorded = guard.record_evidence_request(
        {"query": "FR36 焊接案例", "reason": "缺同类案例", "scope": "cases", "effective_round": 3},
        state=state,
    )

    assert verdict.action == "correct"
    assert recorded.effective_round == 4


def test_an_incomplete_evidence_request_is_rejected() -> None:
    guard, _ = _guard()
    verdict = guard.judge(
        ActionProposal(action="request_evidence", payload={"query": "", "reason": "", "scope": "cases"}),
        LoopState(round_no=2),
    )
    assert verdict.action == "reject"
    assert verdict.reason == "evidence_request_incomplete"


def test_attribution_is_deterministic_and_needs_a_previous_round() -> None:
    """§5.2.6：重叠度由守卫确定性算出（Q-22：只记录，不驱动补救路径）。"""
    guard, _ = _guard(evidence=("E-aaa1", "E-bbb2"))
    state = LoopState(round_no=1)
    assert (
        guard.judge(ActionProposal(action="attribute", payload={}), state).reason
        == "attribute_before_second_round"
    )

    state = LoopState(
        round_no=2,
        latest={
            "E01": _opinion("E01", "approve", ["E-aaa1"]),
            "E02": _opinion("E02", "reject", ["E-bbb2"]),
        },
    )
    verdict = guard.judge(ActionProposal(action="attribute", payload={}), state)
    first = guard.attribute(state=state)
    second = guard.attribute(state=state)

    assert verdict.action == "accept"
    assert first == second, "归因必须是确定性函数"
    assert first.evidence_overlap == 0.0
    assert first.kind in {"evidence", "mixed"}


def test_ask_human_is_allowed_once_per_run() -> None:
    guard, _ = _guard()
    state = LoopState(round_no=2)
    proposal = ActionProposal(action="ask_human", payload={"reason": "consensus_not_reached"})

    first = guard.judge(proposal, state)
    guard.ask_human(proposal.payload, state=state)
    second = guard.judge(proposal, state)

    assert first.action == "accept"
    assert second.action == "reject"
    assert second.reason == "ask_human_already_used"


def test_finalize_rejects_an_unsupported_opinion() -> None:
    """§5.4.1：任一意见无支撑就拒绝收束（转 manual_review）。"""
    guard, _ = _guard()
    state = LoopState(
        round_no=1,
        latest={"E01": _opinion("E01", "approve", ["E-ffffffffffff"])},
    )

    verdict = guard.judge(ActionProposal(action="finalize", payload={}), state)

    assert verdict.action == "reject"
    assert verdict.reason == "unsupported_opinion"


def test_finalize_accepts_a_supported_opinion() -> None:
    guard, ctx = _guard()
    real = ctx.registry.all_ids()[0]
    state = LoopState(round_no=1, latest={"E01": _opinion("E01", "approve", [real])})

    assert guard.judge(ActionProposal(action="finalize", payload={}), state).action == "accept"


# --------------------------------------------------------------------------- #
# 3) ActivationPlan 首次有生产者与消费者
# --------------------------------------------------------------------------- #


def test_activation_plan_has_a_producer_and_a_consumer() -> None:
    """这份契约此前是「零生产者零消费者」的死契约（README v2 的原话）。"""
    guard, ctx = _guard()
    baseline = ctx.registry.all_ids()

    plan = RuleSkeleton().plan(
        request=REQUEST, historical_disciplines=(), baseline_ids=baseline
    )
    assert plan.active_experts == select_experts(REQUEST, ())
    assert set(plan.weights) == set(plan.active_experts)
    assert all(ids == baseline for ids in plan.evidence_scope.values())

    payload = plan_payload(plan)
    assert "rationale" not in payload, "D-76：元智能体的推理永不进子智能体 prompt"
    state = LoopState(round_no=1)
    assert guard.judge(ActionProposal(action="dispatch_experts", payload=payload), state).action == "accept"

    tasks = guard.dispatch(payload, state=state, request=REQUEST)
    assert [t.expert for t in tasks] == plan.active_experts
    assert state.plans and state.plans[0].active_experts == plan.active_experts


# --------------------------------------------------------------------------- #
# 4) StepRecord 与整链路
# --------------------------------------------------------------------------- #


def test_steps_are_recorded_before_they_run_and_paired() -> None:
    """§9.4：`step` 在动作执行前落盘，`step_done` 在执行后落盘（append-only 日志）。"""
    sink = _Sink()
    asyncio.run(run(RunInput(request=REQUEST), _context(sink=sink)))

    names = sink.names()
    assert "step" in names and "step_done" in names
    assert names.count("step") == names.count("step_done")

    starts = [f for name, f in sink.events if name == "step"]
    kinds = {record["kind"] for record in starts}
    assert {"think", "act", "guard", "llm"} <= kinds
    assert any(record["node"] == "guard" for record in starts)

    # 同一节点内步序从 0 起、严格递增，且不会重复。
    per_node: dict[tuple[str, int], list[int]] = {}
    for record in starts:
        per_node.setdefault((record["node"], record["round"]), []).append(record["step"])
    for steps in per_node.values():
        assert steps == list(range(len(steps)))


def test_a_run_reports_the_guarded_acts() -> None:
    """一次真实离线 run 里，守卫对每个 act 都有裁定记录。

    用「全票 revise」的脚本化假模型 + 抬高阈值，保证至少跑两轮——否则第 1 轮就被
    批准收束，`attribute` / `read_memory` 这两个只从第 2 轮起出现的 act 根本不会发生。
    """
    sink = _Sink()
    steady = FakeLLM({r: {e: "revise" for e in EXPERT_IDS} for r in (1, 2, 3)})
    result = asyncio.run(
        run(
            RunInput(request=REQUEST),
            _context(sink=sink, llm=steady),
            max_rounds=3,
            threshold=0.9,
        )
    )

    verdicts = [f for name, f in sink.events if name == "guard_verdict"]
    acts = {v["action"] for v in verdicts}
    assert {"dispatch_experts", "attribute", "read_memory", "finalize"} <= acts
    assert all(v["verdict"] in {"accept", "correct", "reject"} for v in verdicts)
    assert result.consensus_status in {
        "approved",
        "manual_review",
        "stalled",
    }
    assert result.active_experts == select_experts(REQUEST, ())


def test_meta_rationale_never_reaches_a_sub_agent_prompt() -> None:
    """D-76：`ActivationPlan.rationale` 只进事件日志与审计，永不进子智能体 prompt。"""
    prompts: list[str] = []

    class _RecordingLLM(FakeLLM):
        async def complete(self, **kwargs: object):  # type: ignore[override]
            prompts.append(str(kwargs.get("user", "")))
            return await super().complete(**kwargs)  # type: ignore[arg-type]

    asyncio.run(run(RunInput(request=REQUEST), _context(llm=_RecordingLLM())))

    assert prompts, "没捕获到任何 prompt，这条用例失去意义"
    assert not any("规则骨架" in prompt for prompt in prompts)


def test_an_evidence_request_takes_effect_in_the_next_round() -> None:
    """D-16 / D-97：请求只能在**下一轮**生效，且在被回填之前一直处于「待满足」。

    此前这条用例的前提是「检索端口没有 `search()`」——那既不是「请求被满足」的判据，也把
    「没去查」与「查了没有」混为一谈。缺口回填（D-97）之后，判据落在请求自己的字段上。
    """
    guard, _ctx = _guard()
    state = LoopState(round_no=2)
    request = guard.record_evidence_request(
        {"query": "FR36 焊接案例", "reason": "缺同类案例", "scope": "cases", "effective_round": 3},
        state=state,
    )

    assert request.effective_round == 3
    assert guard.evidence_requests[0] is request, "重复登记必须返回同一个对象，否则状态会分叉"
    assert guard.due_evidence_requests(2) == [], "还没到生效轮次"
    assert guard.due_evidence_requests(3) == [request]

    guard.mark_evidence_request_satisfied(request, round_no=3, hits=2, new_ids=["E-aaa"])
    assert request.satisfied_round == 3
    assert request.satisfied_hits == 2
    assert request.satisfied_evidence == ["E-aaa"]
    assert guard.due_evidence_requests(4) == [], "已尝试过的不再重复取（幂等，省一次检索）"
