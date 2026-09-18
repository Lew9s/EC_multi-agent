"""D-93 / D-94 / **D-96**：弃权的两种来源、证据缺口、以及「交人原因」。

这个文件钉住四件事：

1. **判断性弃权是一次交付**（算「有效专家」），只有执行失败才是缺席 —— quorum 只该拦后者；
2. 弃权来源由**服务端**判定，模型无法把自己伪装成缺席或反之；
3. 全员判断性弃权 → `manual_review` + `review_reason="evidence_gap"`，且 `assurance`
   **不得**声称 `history_backed`；
4. 证据缺口由专家写好的待补清单**确定性派生**，模型原文永不进 query。
"""

from __future__ import annotations

import asyncio
import json
import re

from ec_renew.agents.experts import select_experts
from ec_renew.agents.guard import Guard, LoopState
from ec_renew.agents.memory import EvidenceRegistry, MemoryService
from ec_renew.agents.meta import MAX_GAP_KEYWORDS, MAX_GAP_REQUESTS, RuleSkeleton
from ec_renew.agents.skills.expert_review import abstain_opinion, parse_opinion
from ec_renew.contracts import (
    EXPERT_IDS,
    ExpertOpinion,
    LLMCallMeta,
    LLMResult,
    RunInput,
    Usage,
)
from ec_renew.llm import FakeLLM
from ec_renew.observability import NullEventLog
from ec_renew.ports import RunContext
from ec_renew.rag import InMemoryRetriever
from ec_renew.workflow import consensus, manual_review_reason, run

EID = "E-9c4b1e7a2f03"
REQUEST = "301分段FR36污水井更换加厚板，涉及焊接，需确认合规性"


def _judgment_abstain(expert: str, uncertainties: list[str] | None = None) -> ExpertOpinion:
    return ExpertOpinion(
        expert=expert,
        decision="abstain",
        abstain_kind="judgment",
        evidence_ids=[EID],
        uncertainties=uncertainties or [],
    )


# --------------------------------------------------------------------------- #
# 1) 「有效专家」= 交付了意见的专家
# --------------------------------------------------------------------------- #


def test_judgment_abstain_is_a_delivery_not_an_absence() -> None:
    """四位专家一致说「证据不足」——他们**都到位了**，这不是缺席。"""
    weights = {expert: 0.25 for expert in EXPERT_IDS[:4]}
    opinions = {expert: _judgment_abstain(expert) for expert in weights}

    score, effective, status = consensus(opinions, weights, 0.6, min_effective=3)

    assert effective == 4, "判断性弃权必须计入「已交付」"
    assert score == 0.0
    # 终态是「交人」，原因才是「证据缺口」——D-96 把这两件事拆开。
    assert status == "manual_review"
    assert manual_review_reason(opinions, min_effective=3) == "evidence_gap"


def test_execution_failure_abstain_is_still_an_absence() -> None:
    """服务端兜底的弃权（超时/传输/契约耗尽）才算缺席，quorum 照旧拦下。"""
    weights = {expert: 1 / 3 for expert in EXPERT_IDS[:3]}
    opinions = {
        "E01": ExpertOpinion(expert="E01", decision="approve", evidence_ids=[EID]),
        "E02": abstain_opinion("E02", [EID], "调用超时"),
        "E03": abstain_opinion("E03", [EID], "传输失败"),
    }
    assert {op.abstain_kind for op in opinions.values() if op.decision == "abstain"} == {
        "execution_failure"
    }

    _, effective, status = consensus(opinions, weights, 0.6, min_effective=3)

    assert effective == 1
    assert status == "manual_review"


def test_a_mixed_round_is_not_an_evidence_gap() -> None:
    """有人给了判断 → 走常规判据（按分数判），不是证据缺口。"""
    weights = {expert: 1 / 3 for expert in EXPERT_IDS[:3]}
    opinions = {
        "E01": ExpertOpinion(expert="E01", decision="revise", evidence_ids=[EID]),
        "E02": _judgment_abstain("E02"),
        "E03": _judgment_abstain("E03"),
    }

    score, effective, status = consensus(opinions, weights, 0.6, min_effective=3)

    assert effective == 3
    assert score == 0.0 or score > 0.0
    assert status == "retry"


# --------------------------------------------------------------------------- #
# 2) 弃权来源由服务端判定（模型不可伪造）
# --------------------------------------------------------------------------- #


def test_the_model_cannot_forge_the_abstain_source() -> None:
    """模型 JSON 里写 `abstain_kind` 不采信——它是 quorum 的判据（D-93）。"""
    forged = json.dumps(
        {
            "decision": "abstain",
            "abstain_kind": "execution_failure",  # 试图把自己伪装成「缺席」
            "evidence_ids": [EID],
        }
    )
    opinion = parse_opinion("E01", forged, [EID])
    assert opinion.abstain_kind == "judgment"

    normal = parse_opinion("E01", json.dumps({"decision": "approve", "evidence_ids": [EID]}), [EID])
    assert normal.abstain_kind is None, "有明确决策时不该带弃权来源"


# --------------------------------------------------------------------------- #
# 3) 缺口派生（D-94）
# --------------------------------------------------------------------------- #


def _state_with_shortfalls(*uncertainties: str) -> LoopState:
    return LoopState(
        round_no=2,
        latest={"E01": _judgment_abstain("E01", list(uncertainties))},
    )


def test_gaps_are_derived_from_the_declared_shortfalls() -> None:
    """按固定词表映射到 scope；query 用「原始请求 + 词表词」拼，**不含模型原文**。"""
    secret = "某位专家写的很特别的原话ABC"
    state = _state_with_shortfalls(
        f"未提供焊接工艺文件（WPS）与探伤比例，{secret}",
        "缺少船级社认可意见",
    )

    gaps = RuleSkeleton().gaps(state, request=REQUEST)

    assert [gap.scope for gap in gaps] == ["standards"]
    query = gaps[0].query
    assert REQUEST in query, "原始请求必须在 query 里（检索要有上下文）"
    assert "wps" in query
    assert secret not in query, "模型原文绝不进 query（否则等于让概率文本驱动检索）"
    assert gaps[0].effective_round == 3, "D-16：只能对下一轮生效"


def test_gap_derivation_is_capped() -> None:
    state = _state_with_shortfalls(
        "规范 标准 船级社 证书 WPS 探伤 板厚 材质 图纸 部门 签收 船东 同类案例"
    )

    gaps = RuleSkeleton().gaps(state, request=REQUEST)

    assert len(gaps) <= MAX_GAP_REQUESTS
    for gap in gaps:
        assert gap.query.count("、") + 1 <= MAX_GAP_KEYWORDS


def test_no_shortfalls_means_no_requests() -> None:
    assert RuleSkeleton().gaps(_state_with_shortfalls(), request=REQUEST) == []


def test_recording_the_same_gap_twice_is_idempotent() -> None:
    """缺口每轮都会重新派生 → 记账必须幂等，否则「提了几次」失去意义。"""
    registry = EvidenceRegistry()
    registry.register(source="graph", content="case A")
    registry.freeze(1, registry.all_ids())
    ctx = RunContext(
        run_id="gap",
        llm=FakeLLM(),
        registry=registry,
        events=NullEventLog(),
        retriever=None,
    )
    guard = Guard(ctx, memory=MemoryService(registry))
    payload = {
        "query": f"{REQUEST}｜需补充：wps",
        "reason": "缺少 WPS",
        "scope": "standards",
        "effective_round": 2,
    }

    guard.record_evidence_request(payload, state=LoopState(round_no=1))
    guard.record_evidence_request(payload, state=LoopState(round_no=2))

    assert len(guard.evidence_requests) == 1


# --------------------------------------------------------------------------- #
# 4) 端到端：全员判断性弃权
# --------------------------------------------------------------------------- #


class _AbstainingLLM:
    """每次都给「判断性弃权 + 待补清单」：全员弃权的形态。"""

    async def complete(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        meta: LLMCallMeta | None = None,
    ) -> LLMResult:
        cited = sorted(set(re.findall(r"E-[0-9a-f]{12}", user))) or [EID]
        return LLMResult(
            content=json.dumps(
                {
                    "decision": "abstain",
                    "basis": "evidence",
                    "rationale": "本轮证据不含技术参数，无法给出结论",
                    "evidence_ids": cited,
                    "uncertainties": ["缺少焊接工艺文件（WPS）与探伤比例", "板厚与材质未给出"],
                    "risk_level": "high",
                },
                ensure_ascii=False,
            ),
            model="abstaining",
            usage=Usage(calls=1),
        )


def test_an_all_abstain_run_reports_the_gap_instead_of_absent_experts() -> None:
    """全员判断性弃权的终态是 `manual_review`，但**原因**是证据缺口而不是专家缺席，
    并且要交出缺口清单（D-93 的分类 + D-96 的原因标签）。"""
    ctx = RunContext(
        run_id="all-abstain",
        llm=_AbstainingLLM(),  # type: ignore[arg-type]
        registry=EvidenceRegistry(),
        events=NullEventLog(),
        retriever=InMemoryRetriever(),  # 有历史命中，好检验 assurance 的降级
    )

    result = asyncio.run(run(RunInput(request=REQUEST), ctx))

    assert result.consensus_status == "manual_review"
    assert result.review_reason == "evidence_gap"
    assert result.rounds == 1, "证据缺口不该硬迭代（D-29 / §5.5.5）"
    assert "max_rounds_reached" not in result.warnings
    assert result.evidence_requests, "缺口必须交出去（结构化契约，不只是渲染文本）"
    assert "需要补充的证据" in result.conclusion
    assert "evidence_request_unsatisfied" in result.warnings, "检索端未实现要显式说明（P5）"
    assert result.assurance.level == "knowledge_based", (
        "本轮没有任何专业判断：不得声称 history_backed"
    )
    assert result.active_experts == select_experts(REQUEST, [])
