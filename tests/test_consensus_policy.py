"""D-95 / **D-96**：共识判定分两层，而「停止时如何分类」只分**交付 / 交人**。

本文件把优化前后的差别钉成一张**可复算的矩阵**。优化前 `score ≥ threshold` 一个标量决定一切，
于是出现了三件事：

1. **二义性**：`score = 0.5 + (a − j − x)/2` 里 `revise` 被消掉 → 「全员一致附条件」与
   「赞反对峙」算出同一个 `0.5000`，状态机也报同一个 `manual_review`；
2. **n 敏感**：「1 位赞成 + 其余仅提条件」在 n=5 时恰好 0.60 通过、n=6 时 0.5833 不通过
   —— 同样内容，只因多请了一位只提条件的专家就翻结论；
3. **全 `revise` 恒 0.50**：结构性不可能通过，且被误报成「专家分歧未解决」。

D-95 把**收敛判据**改成结构化的（无 `reject` + 至少一位 `approve` + 支持度 ≥ threshold），
并给终态补了 `conditional`。D-96 再把终态收成两个出口（交付 / 交人），把「因为什么交人」
移到 `review_reason`。

**本文件同时守着这两条**：状态值不再区分「附条件」与「分歧」（`test_a_deadlock_...`），
但这两者**没有被丢掉**——它们在 `manual_review_reason` 里分得开（`REASON_MATRIX`）。
"""

from __future__ import annotations

import asyncio

import pytest

from ec_renew.agents.memory import EvidenceRegistry
from ec_renew.agents.skills.expert_review import abstain_opinion
from ec_renew.contracts import EXPERT_IDS, ExpertOpinion, RunInput
from ec_renew.observability import NullEventLog
from ec_renew.ports import RunContext
from ec_renew.workflow import consensus, manual_review_reason, run

EID = "E-9c4b1e7a2f03"
THRESHOLD = 0.6
MIN_EFFECTIVE = 3


def _op(expert: str, decision: str) -> ExpertOpinion:
    return ExpertOpinion(expert=expert, decision=decision, evidence_ids=[EID])  # type: ignore[arg-type]


def _panel(decisions: list[str]) -> tuple[dict[str, ExpertOpinion], dict[str, float]]:
    experts = [f"E{i:02d}" for i in range(1, len(decisions) + 1)]
    return {e: _op(e, d) for e, d in zip(experts, decisions)}, {e: 1.0 for e in experts}


def _judge(decisions: list[str], *, final: bool) -> str:
    opinions, weights = _panel(decisions)
    _score, _effective, status = consensus(
        opinions, weights, THRESHOLD, MIN_EFFECTIVE, final=final
    )
    return status


def _reason(decisions: list[str]) -> str:
    opinions, _weights = _panel(decisions)
    return manual_review_reason(opinions, MIN_EFFECTIVE)


# --------------------------------------------------------------------------- #
# 第 2 层：停止时如何分类（final=True）—— 只有「交付」与「交人」两个出口
# --------------------------------------------------------------------------- #

MATRIX: tuple[tuple[str, list[str], str], ...] = (
    # 全 revise：一致认为方向可行、需先满足条件 —— 无人反对，但没人明确赞成，所以**交人**
    ("全 revise（n=3）", ["revise"] * 3, "manual_review"),
    ("全 revise（n=6）", ["revise"] * 6, "manual_review"),
    # 有人明确反对 → 交人裁定。注意这**不再**被「赞成票更多」平均掉
    ("半 approve 半 reject（n=4）", ["approve", "approve", "reject", "reject"], "manual_review"),
    ("3 approve + 1 reject（n=4）", ["approve"] * 3 + ["reject"], "manual_review"),
    ("2 approve + 2 revise + 2 reject", ["approve"] * 2 + ["revise"] * 2 + ["reject"] * 2, "manual_review"),
    # 无反对 + 有明确赞成 + 支持度达标 → 通过（与激活了几位专家无关）
    ("1 approve + 其余 revise（n=3）", ["approve", "revise", "revise"], "approved"),
    ("1 approve + 其余 revise（n=5）", ["approve"] + ["revise"] * 4, "approved"),
    # 支持度 1/6 = 0.167 < 0.6 → 交人，而不是「通过」：五位专家都只肯说「先满足条件」，
    # 标成 approved 会高估支持度。**残余的 n 敏感性就留在这一档**（approved↔manual_review），
    # 它现在不影响「交给谁」，只影响「明确支持够不够」；口径见 Q-26。
    ("1 approve + 其余 revise（n=6）", ["approve"] + ["revise"] * 5, "manual_review"),
    ("全 approve（n=4）", ["approve"] * 4, "approved"),
    # 支持度不足（1 赞成 + 4 弃权 → 0.2）→ 交人，而不是「通过」
    ("1 approve + 4 判断性弃权", ["approve"] + ["abstain"] * 4, "manual_review"),
)


@pytest.mark.parametrize(("label", "decisions", "expected"), MATRIX, ids=[m[0] for m in MATRIX])
def test_final_classification(label: str, decisions: list[str], expected: str) -> None:
    assert _judge(decisions, final=True) == expected


def test_state_values_are_exactly_three() -> None:
    """状态机只有三个取值（D-96）。这条不是形式主义：每多一个取值，每个消费者就多一个必须
    分支的取值，而这两个被并掉的取值**没有对应的行为差**（收束 act 只认 approved / stalled，
    `conditional` 从未因此自动交付，它一直也在问人）。"""
    observed = {terminal for _label, _decisions, terminal in MATRIX}
    assert observed == {"approved", "manual_review"}


# --------------------------------------------------------------------------- #
# 被并掉的两个取值：信息搬到了 `review_reason`，**没有丢**
# --------------------------------------------------------------------------- #

REASON_MATRIX: tuple[tuple[str, list[str], str], ...] = (
    # 缺席（有效专家不足）—— D-37 一直要求「缺席不得当作通过」，却从没给它一个可辨认的标签
    ("全执行失败弃权（n=3）", ["abstain"] * 3, "quorum"),
    ("全员判断性弃权", ["abstain"] * 3, "evidence_gap"),
    ("有人明确反对", ["approve", "approve", "reject", "reject"], "disagreement"),
    ("全 revise（附条件）", ["revise"] * 3, "conditions_only"),
    ("1 approve + 其余 revise（n=6）", ["approve"] + ["revise"] * 5, "conditions_only"),
)


@pytest.mark.parametrize(("label", "decisions", "expected"), REASON_MATRIX, ids=[m[0] for m in REASON_MATRIX])
def test_manual_review_reason(label: str, decisions: list[str], expected: str) -> None:
    """原因判定。`quorum` 与 `evidence_gap` 都长着弃权的样子，靠 `abstain_kind` 分开。"""
    opinions, _weights = _panel(decisions)
    if expected == "quorum":  # 执行失败弃权 = 缺席（不是交付）
        for op in opinions.values():
            op.abstain_kind = "execution_failure"
    elif expected == "evidence_gap":  # 判断性弃权 = 交付，只是交付的内容是「证据不足」
        for op in opinions.values():
            op.abstain_kind = "judgment"
    assert manual_review_reason(opinions, MIN_EFFECTIVE) == expected


def test_all_revise_is_the_same_verdict_for_any_panel_size() -> None:
    """优化前：n=3 与 n=6 都是 0.5000（二义性）；现在两者都明确交人，原因同为「附条件」。"""
    assert _judge(["revise"] * 3, final=True) == "manual_review"
    assert _judge(["revise"] * 6, final=True) == "manual_review"
    assert _reason(["revise"] * 3) == _reason(["revise"] * 6) == "conditions_only"


def test_one_approver_no_longer_flips_with_panel_size() -> None:
    """优化前：n=5 恰好压线 0.6000 通过、**n=6 只有 0.5833 不通过**——同样内容两种结论，
    而且「不通过」意味着交人裁定。

    优化后：两种规模都只看「有没有人反对」（与规模无关）；差别只剩「明确支持够不够」这一档：
    n=5 支持度达标 → 交付，n=6 → 交人附条件确认。残余的规模敏感性被限制在**交付强度**上，
    见 Q-26。
    """
    assert _judge(["approve"] + ["revise"] * 4, final=True) == "approved"
    assert _judge(["approve"] + ["revise"] * 5, final=True) == "manual_review"


def test_a_deadlock_is_not_reported_as_agreement() -> None:
    """D-95 的原始用例：优化前「半赞成半反对」与「全 revise」同为 0.5000，无法区分。

    D-96 之后两者的**状态值**重新变成同一个 `manual_review`——但区分还在，只是搬到了原因上。
    这正是本条用例要守的东西：并掉状态取值**不等于**并掉那条信息。
    """
    assert _judge(["approve", "approve", "reject", "reject"], final=True) == "manual_review"
    assert _judge(["revise"] * 4, final=True) == "manual_review"
    assert _reason(["approve", "approve", "reject", "reject"]) == "disagreement"
    assert _reason(["revise"] * 4) == "conditions_only"


# --------------------------------------------------------------------------- #
# 第 1 层：是否收敛（决定要不要继续迭代）
# --------------------------------------------------------------------------- #


def test_unconverged_rounds_still_retry() -> None:
    """`final=False` 时未收敛一律 `retry`——迭代不能被这次优化掐掉。"""
    for _label, decisions, terminal in MATRIX:
        if terminal != "approved":
            assert _judge(decisions, final=False) == "retry"


def test_converged_rounds_report_approved_immediately() -> None:
    assert _judge(["approve"] * 4, final=False) == "approved"


def test_quorum_and_evidence_gap_are_terminal_at_once() -> None:
    """缺席与「一致判断证据不足」不等待轮次：它们不是「再跑一轮就能变好」的情形。

    D-96 之后两者都是 `manual_review`（当轮收束），区分靠原因——如果只看状态值，这两个**补救
    动作相反**的情形（补评审 vs 补证据）会长得一模一样。
    """
    experts = ["E01", "E02", "E03"]
    weights = {e: 1.0 for e in experts}
    absent = {
        "E01": _op("E01", "approve"),
        "E02": abstain_opinion("E02", [EID], "调用超时"),
        "E03": abstain_opinion("E03", [EID], "传输失败"),
    }
    all_gap = {e: _op(e, "abstain") for e in experts}
    for opinion in all_gap.values():
        opinion.abstain_kind = "judgment"

    assert consensus(absent, weights, THRESHOLD, MIN_EFFECTIVE, final=False)[2] == "manual_review"
    assert consensus(all_gap, weights, THRESHOLD, MIN_EFFECTIVE, final=False)[2] == "manual_review"
    assert manual_review_reason(absent, MIN_EFFECTIVE) == "quorum"
    assert manual_review_reason(all_gap, MIN_EFFECTIVE) == "evidence_gap"


# --------------------------------------------------------------------------- #
# 前置条件与交人原因必须进入交付物
# --------------------------------------------------------------------------- #


class _ScriptedLLM:
    """按 decision 脚本作答；`revise` 时各自列出前置条件，复现「方向可行、条件待满足」。"""

    def __init__(self, decision: str) -> None:
        self._decision = decision

    async def complete(self, *, purpose: str, system: str, user: str, meta=None):  # type: ignore[no-untyped-def]
        import json
        import re

        from ec_renew.contracts import LLMCallMeta, LLMResult, Usage

        cited = sorted(set(re.findall(r"E-[0-9a-f]{12}", user))) or [EID]
        content = json.dumps(
            {
                "decision": self._decision,
                "basis": "mixed",
                "rationale": "方向可行，须先补齐焊接工艺文件",
                "evidence_ids": cited,
                "constraints": ["须提交经船级社认可的 WPS/PQR", "焊缝须明确探伤比例与验收标准"],
                "uncertainties": ["板厚与材质未给出"],
                "risk_level": "medium",
            },
            ensure_ascii=False,
        )
        assert isinstance(meta, LLMCallMeta | None)
        return LLMResult(content=content, model=self._decision, usage=Usage(calls=1))


def _run_offline(llm: object, run_id: str, *, max_rounds: int = 1):  # type: ignore[no-untyped-def]
    ctx = RunContext(
        run_id=run_id,
        llm=llm,  # type: ignore[arg-type]
        registry=EvidenceRegistry(),
        events=NullEventLog(),
        retriever=None,
    )
    # max_rounds=1：恒定的脚本化模型会在第 2 轮正确触发**不动点**（无新证据 + 零变化），
    # 那会得到 `stalled`；本用例要验证的是「一轮结束、无人反对 → 交人附条件」。
    return asyncio.run(run(RunInput(request="301分段FR36污水井更换加厚板"), ctx, max_rounds=max_rounds))


def test_conditions_travel_with_the_deliverable() -> None:
    """终态是「交人 + 附条件」，前置条件与交人原因进入**结构化契约**与报告，而不只是散文。"""
    result = _run_offline(_ScriptedLLM("revise"), "conditions-only")

    assert result.consensus_status == "manual_review"
    assert result.review_reason == "conditions_only"
    assert result.conditions, "前置条件必须结构化地跟着结论走"
    assert any("WPS" in item for item in result.conditions)
    assert "前置条件（施工/采购前必须满足）" in result.conclusion
    assert "交人原因" in result.conclusion, "接手的人要知道**因为什么**交人（D-96）"
    assert "描述性统计" in result.conclusion, "支持度不再是判据，报告里要说清"
    assert result.active_experts == [e for e in EXPERT_IDS if e in set(result.active_experts)]


def test_an_approved_run_carries_no_review_reason() -> None:
    """交付不是「交人」，所以**没有**原因标签——`None` 与「原因未知」必须分得开。"""
    result = _run_offline(_ScriptedLLM("approve"), "approved")

    assert result.consensus_status == "approved"
    assert result.review_reason is None
    assert "交人原因" not in result.conclusion
