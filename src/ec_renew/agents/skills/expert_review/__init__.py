"""专家评审技能：让一个 agent 对工程变更给出**可回溯、结构化**的专业评审意见。

完整的能力说明见同目录 ``SKILL.md``；该技能的用例见 ``tests/test_skill_expert_review.py``。
"""

from __future__ import annotations

from .. import SkillSpec
from .personas import ROLE_PROMPTS
from .prompt import SYSTEM_TEMPLATE, system_for
from .runner import (
    abstain_opinion,
    extract_json,
    parse_opinion,
    render_task,
    repair_opinion,
    run_expert,
)

SKILL = SkillSpec(
    name="expert_review",
    version="1.0.0",
    description="按专业 persona 对工程变更给出结构化评审意见（含依据声明、契约校验与重试）",
    inputs="ExpertTask",
    outputs="ExpertOpinion",
    tools=("search_evidence",),
)

__all__ = [
    "ROLE_PROMPTS",
    "SKILL",
    "SYSTEM_TEMPLATE",
    "abstain_opinion",
    "extract_json",
    "parse_opinion",
    "render_task",
    "repair_opinion",
    "run_expert",
    "system_for",
]
