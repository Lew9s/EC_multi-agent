"""**D-98**：方案生成节点 —— 框架闭环的最后一环。

这个文件钉住五件事：

1. **可交付终态才有方案**：`approved` 产出方案；`manual_review` **不产出**（它走的是「回中段再跑」，
   不是交付路径，D-99）；
2. **降级显式**：没有可用模型 / 契约重试耗尽 → 确定性模板，`plan_source="template"`（P5）；
3. **必须章节无支撑 → 不得交付**：拒绝整份方案、降 `manual_review`、`review_reason="unsupported_plan"`
   （§5.7 硬约束）；
4. **`conditions` 不能丢**：方案里没写到的前置条件由守卫**补条**（D-95：否则「有条件通过」在交付物
   里变成「无条件通过」）；
5. **逐条可回溯**：引用了本轮 registry 之外的 `evidence_id` 的条目被剔除。
"""

from __future__ import annotations

import asyncio
import json
import re

from ec_renew.agents.memory import EvidenceRegistry
from ec_renew.contracts import (
    REQUIRED_PLAN_HEADINGS,
    LLMCallMeta,
    LLMResult,
    RunInput,
    SessionSnapshot,
    Usage,
)
from ec_renew.observability import NullEventLog
from ec_renew.ports import RunContext
from ec_renew.rag import InMemoryRetriever
from ec_renew.workflow import run

REQUEST = "301分段FR36污水井更换加厚板，涉及焊接，需确认合规性"
ALL_HEADINGS = ("scope", "basis", "execution", "risk", "open_questions", "evidence_index")
CONDITION = "须提交经船级社认可的 WPS/PQR"


def _plan_json(
    ids: list[str], *, empty: tuple[str, ...] = (), bogus: bool = False, with_condition: bool = False
) -> str:
    """构造一份方案。``empty`` 里的章节**故意不给条目**；``bogus`` 时每节各带一条越界引用的条目
    （另有一条合法条目，用来验证「剔除越界条」不会连带作废整节）。"""
    fallback = ids[0] if ids else "E-000000000000"
    sections = []
    for key in ALL_HEADINGS:
        if key in empty:
            sections.append({"heading": key, "body": f"{key} 叙述", "claims": []})
            continue
        claims = [{"text": f"{key} 条目", "evidence_ids": [fallback]}]
        if bogus:
            claims.append({"text": f"{key} 越界条目", "evidence_ids": ["E-000000000000"]})
        if with_condition and key == "risk":
            claims.append({"text": f"施工前必须满足：{CONDITION}", "evidence_ids": [fallback]})
        sections.append({"heading": key, "body": f"{key} 的叙述正文", "claims": claims})
    return json.dumps(
        {"sections": sections, "assumption_notes": ["板厚按原图纸 12mm 计，未获确认"]},
        ensure_ascii=False,
    )


class _ScriptedLLM:
    """按 ``purpose`` 分流：专家回意见（可指定决策），方案节点回方案 JSON。

    方案里的证据 id 取自**本次 prompt 里真实出现的 id**——写死的 id 会让「越界剔除」这条判据
    在测试里失效（所有条目都被剔掉，看起来像「必需章节无支撑」）。
    """

    def __init__(
        self,
        *,
        decision: str = "approve",
        empty: tuple[str, ...] = (),
        bogus: bool = False,
        with_condition: bool = False,
        broken_plan: bool = False,
    ) -> None:
        self._decision = decision
        self._empty = empty
        self._bogus = bogus
        self._with_condition = with_condition
        self._broken = broken_plan
        self.prompts: list[tuple[str, str]] = []

    async def complete(
        self, *, purpose: str, system: str, user: str, meta: LLMCallMeta | None = None
    ) -> LLMResult:
        self.prompts.append((purpose, user))
        ids = sorted(set(re.findall(r"E-[0-9a-f]{12}", user)))
        if purpose == "plan":
            if self._broken:
                return LLMResult(content="这不是 JSON", model="plan", usage=Usage(calls=1))
            return LLMResult(
                content=_plan_json(
                    ids,
                    empty=self._empty,
                    bogus=self._bogus,
                    with_condition=self._with_condition,
                ),
                model="plan",
                usage=Usage(calls=1),
            )
        return LLMResult(
            content=json.dumps(
                {
                    "decision": self._decision,
                    "basis": "mixed",
                    "rationale": "r",
                    "evidence_ids": ids or ["E-9c4b1e7a2f03"],
                    "constraints": [CONDITION],
                    "uncertainties": ["板厚与材质未给出"],
                    "risk_level": "medium",
                },
                ensure_ascii=False,
            ),
            model="expert",
            usage=Usage(calls=1),
        )


def _run(
    llm: object,
    *,
    session: SessionSnapshot | None = None,
    plan_feedback: list[str] | None = None,
    run_id: str = "plan",
):  # type: ignore[no-untyped-def]
    ctx = RunContext(
        run_id=run_id,
        llm=llm,  # type: ignore[arg-type]
        registry=EvidenceRegistry(),
        events=NullEventLog(),
        retriever=InMemoryRetriever(),
    )
    run_input = RunInput(
        request=REQUEST,
        session=session or SessionSnapshot(session_id="s", anchor_request=REQUEST),
        plan_feedback=plan_feedback or [],
    )
    return asyncio.run(run(run_input, ctx))


# --------------------------------------------------------------------------- #
# 1) 可交付终态产出方案；交人不产出
# --------------------------------------------------------------------------- #


def test_an_approved_run_produces_a_plan_from_the_agent() -> None:
    result = _run(_ScriptedLLM(decision="approve"))

    assert result.consensus_status == "approved"
    assert result.plan is not None
    assert result.plan_source == "agent"
    assert {section.heading for section in result.plan.sections} == set(ALL_HEADINGS)
    # 交付物文本里章节标题是可读的，条目带证据引用。
    assert "# 变更方案" in result.plan_markdown
    assert "变更范围" in result.plan_markdown
    assert "（依据 E-" in result.plan_markdown
    # 假设必须被看见，且与条目分开。
    assert "撰写假设" in result.plan_markdown


def test_a_manual_review_run_gets_no_plan() -> None:
    """交人不是交付路径（D-99）：人补的是证据与专家分配，不是一份半成品方案。

    全员 `reject` 且意见逐轮不变时，流程会先在不动点上报 `stalled`（§5.4.4）——两者都不是
    `approved`，因此都不该出方案；这里断言的是这条共同性质，而不是某一个具体终态。
    """
    result = _run(_ScriptedLLM(decision="reject"), run_id="plan-manual")

    assert result.consensus_status != "approved"
    assert result.plan is None
    assert result.plan_source is None
    assert result.plan_markdown == ""
    assert any(item.startswith("plan_skipped:") for item in result.warnings)


# --------------------------------------------------------------------------- #
# 2) 降级必须显式
# --------------------------------------------------------------------------- #


def test_a_broken_plan_contract_falls_back_to_the_deterministic_template() -> None:
    result = _run(_ScriptedLLM(decision="approve", broken_plan=True), run_id="plan-fallback")

    assert result.plan_source == "template", "没有可用方案时必须回落，且要说出来"
    assert "plan_agent_failed" in result.warnings
    assert "plan_template_fallback" in result.warnings
    assert "确定性模板" in result.plan_markdown
    assert result.plan is not None
    assert REQUIRED_PLAN_HEADINGS <= {section.heading for section in result.plan.sections}


def test_the_template_fallback_still_carries_every_condition() -> None:
    """回落路径不能成为「条件丢失」的后门（D-95）。"""
    result = _run(_ScriptedLLM(decision="approve", broken_plan=True), run_id="plan-fallback-2")

    assert CONDITION in result.plan_markdown
    assert CONDITION in result.conditions


# --------------------------------------------------------------------------- #
# 3) §5.7 硬约束：必需章节无支撑 → 不得交付
# --------------------------------------------------------------------------- #


def test_a_plan_with_an_unsupported_required_section_is_rejected() -> None:
    result = _run(_ScriptedLLM(decision="approve", empty=("basis",)), run_id="plan-unsupported")

    assert result.plan is None, "必需章节没有一条有支撑 ⇒ 整份方案不得交付"
    assert result.consensus_status == "manual_review"
    assert result.review_reason == "unsupported_plan"
    assert any(item.startswith("plan_unsupported:") for item in result.warnings)
    # 报告必须说清为什么交人（D-96 的「交人原因」行）。
    assert "交人原因" in result.conclusion


# --------------------------------------------------------------------------- #
# 4) 逐条可回溯 + 条件不丢
# --------------------------------------------------------------------------- #


def test_a_claim_citing_an_unknown_evidence_id_is_dropped_not_the_section() -> None:
    result = _run(_ScriptedLLM(decision="approve", bogus=True), run_id="plan-bogus")

    assert result.plan is not None, "只该剔除越界的那条，不该连带作废整节"
    dropped = [item for item in result.warnings if item.startswith("plan:dropped_unsupported")]
    assert dropped, f"越界引用必须被剔除并说出来：{result.warnings}"
    cited = {
        eid
        for section in result.plan.sections
        for claim in section.claims
        for eid in claim.evidence_ids
    }
    assert "E-000000000000" not in cited
    assert all(
        section.claims for section in result.plan.sections if section.heading in REQUIRED_PLAN_HEADINGS
    ), "剔除后必需章节仍应有有效条目"


def test_a_plan_that_omits_a_condition_gets_it_appended() -> None:
    """守卫**补条**而不是拒绝：不变量是「条件必须随交付物走」，补上就满足了它。"""
    plain = _ScriptedLLM(decision="approve")
    result = _run(plain, run_id="plan-conditions")

    assert CONDITION not in _plan_json([]), "前提：模型给的方案里确实没有这条条件"
    assert CONDITION in result.plan_markdown
    assert "plan:appended_conditions:1" in result.warnings


def test_a_plan_that_already_states_the_condition_does_not_get_it_twice() -> None:
    result = _run(
        _ScriptedLLM(decision="approve", with_condition=True), run_id="plan-conditions-dup"
    )

    assert result.plan_markdown.count(CONDITION) == 1, "条件只能出现一次，重复是噪声"
    assert not any("appended_conditions" in item for item in result.warnings)


# --------------------------------------------------------------------------- #
# 5) 迭代回路：方案撰写者拿到上一版与用户意见（D-99 的载体）
# --------------------------------------------------------------------------- #


def test_the_plan_prompt_carries_the_previous_plan_and_the_user_feedback() -> None:
    first = _run(_ScriptedLLM(decision="approve"), run_id="plan-v1")
    assert first.plan is not None

    llm = _ScriptedLLM(decision="approve")
    result = _run(
        llm,
        session=SessionSnapshot(session_id="s", anchor_request=REQUEST, latest_plan=first.plan),
        plan_feedback=["把探伤比例写成 100%"],
        run_id="plan-v2",
    )

    plan_prompt = next(user for purpose, user in llm.prompts if purpose == "plan")
    assert "上一版方案" in plan_prompt, "迭代必须拿得到被改的那一版"
    assert "把探伤比例写成 100%" in plan_prompt, "用户意见必须进方案节点的输入"
    assert result.plan is not None and result.plan.version == 2, "迭代应递增版本号"
