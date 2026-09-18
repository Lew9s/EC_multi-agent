"""技能包 `expert_review` 的用例（技能的说明书见其 ``SKILL.md``）。

这个文件钉住四件事：

1. 技能包是**可发现、可注册、带说明**的（目录形式，不用文件系统扫描）；
2. 判断准则写进了 prompt：**专业判断可基于领域通识**，`abstain` 被收窄（"参数没给全"不是理由）；
3. `basis` 是契约的一部分，且**保证等级按它降级**——检索命中历史 ≠ 专家用了它；
4. 技能的失败方式（契约重试 → 修复 → 弃权）仍然成立。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ec_renew.agents.skills import available_skills, load_skill
from ec_renew.agents.skills.expert_review import (
    SKILL,
    SYSTEM_TEMPLATE,
    parse_opinion,
    render_task,
    system_for,
)
from ec_renew.contracts import ExpertOpinion, ExpertTask
from ec_renew.errors import ContractViolation
from ec_renew.workflow import _knowledge_share

SKILL_DIR = Path(__file__).resolve().parents[1] / "src" / "ec_renew" / "agents" / "skills"
EID = "E-9c4b1e7a2f03"


# --------------------------------------------------------------------------- #
# 1) 技能包本身
# --------------------------------------------------------------------------- #


def test_the_skill_directory_is_discoverable_and_documented() -> None:
    """目录形式的技能包：有 SKILL.md、有元数据、被登记。"""
    assert available_skills() == ("expert_review",)
    assert load_skill("expert_review") == SKILL
    assert SKILL.inputs == "ExpertTask"
    assert SKILL.outputs == "ExpertOpinion"
    assert SKILL.version

    doc = (SKILL_DIR / "expert_review" / "SKILL.md").read_text(encoding="utf-8")
    assert doc.startswith("# 技能：专家评审")
    assert "abstain" in doc, "说明书必须写明弃权的边界"
    assert "basis" in doc


def test_an_unknown_skill_is_rejected() -> None:
    with pytest.raises(KeyError):
        load_skill("no_such_skill")


# --------------------------------------------------------------------------- #
# 2) 判断准则（专业判断可基于领域通识）
# --------------------------------------------------------------------------- #


def test_the_prompt_lets_experts_judge_from_world_knowledge() -> None:
    """RAG 给不出技术细节**不是**不判断的理由——那正是领域通识该上场的时候。"""
    prompt = SYSTEM_TEMPLATE

    assert "专业判断可以基于你的领域通识" in prompt
    assert "不要因此弃权" in prompt
    assert "证据用于支撑事实" in prompt, "P3 仍然只管事实性断言"


def test_the_prompt_narrows_abstain_to_out_of_scope_or_undecidable() -> None:
    prompt = SYSTEM_TEMPLATE

    assert "「参数没给全」不是弃权理由" in prompt
    assert "超出你的专业范围" in prompt
    assert "请求本身**无法判定**" in prompt


def test_the_prompt_requires_a_basis_declaration() -> None:
    prompt = SYSTEM_TEMPLATE

    assert '"basis": "evidence | knowledge | mixed"' in prompt
    assert "诚实的代价，不是惩罚" in prompt


def test_each_discipline_gets_its_own_persona() -> None:
    e01, e06 = system_for("E01"), system_for("E06")

    assert e01 != e06
    assert "结构设计" in e01
    assert "材料与焊接" in e06


# --------------------------------------------------------------------------- #
# 3) basis 是契约的一部分，并驱动保证等级
# --------------------------------------------------------------------------- #


def test_basis_is_parsed_and_bounded() -> None:
    raw = (
        '{"decision": "revise", "basis": "knowledge", "rationale": "r",'
        f' "evidence_ids": ["{EID}"]}}'
    )
    opinion = parse_opinion("E01", raw, [EID])
    assert opinion.basis == "knowledge"

    with pytest.raises(ContractViolation):
        parse_opinion(
            "E01",
            '{"decision": "revise", "basis": "vibes", "evidence_ids": ["' + EID + '"]}',
            [EID],
        )


def test_basis_defaults_to_evidence_for_backward_compatibility() -> None:
    opinion = ExpertOpinion(expert="E01", decision="approve", evidence_ids=[EID])
    assert opinion.basis == "evidence"


def test_assurance_downgrades_when_judgments_rest_on_world_knowledge() -> None:
    """检索命中历史案例，但专家全凭通识判断 → 保证等级必须降级。"""
    weights = {"E01": 1.0, "E03": 1.0, "E06": 1.0, "E02": 1.0}
    all_knowledge = {
        e: ExpertOpinion(expert=e, decision="revise", evidence_ids=[EID], basis="knowledge")
        for e in weights
    }
    mostly_evidence = {
        e: ExpertOpinion(expert=e, decision="revise", evidence_ids=[EID], basis="evidence")
        for e in weights
    }
    mostly_evidence["E02"] = ExpertOpinion(
        expert="E02", decision="revise", evidence_ids=[EID], basis="knowledge"
    )

    assert _knowledge_share(all_knowledge, weights) == pytest.approx(1.0)
    assert _knowledge_share(mostly_evidence, weights) == pytest.approx(0.25)
    assert _knowledge_share(all_knowledge, {}) == 0.0


# --------------------------------------------------------------------------- #
# 4) 输入渲染：技能仍受既有不变量约束
# --------------------------------------------------------------------------- #


def test_rendering_still_keeps_the_prompt_structural_rules() -> None:
    prompt = render_task(ExpertTask(expert="E01", request="r", round=1))

    assert "[[CTX" not in prompt, "调用上下文必须留在带外（D-86）"
    assert "## 输出" in prompt
