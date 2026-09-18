"""方案撰写技能：把**专家意见 + 用户反馈 + 共识终态**写成一份可施工的变更方案（D-98）。

与 `expert_review` 的关系：那个技能回答「这件事能不能做、要注意什么」，这个技能回答「那就怎么做、
依据是什么、前置条件是什么」。两者都是目录形式的技能包，都被同一份 runtime 执行。

完整能力说明见同目录 ``SKILL.md``。
"""

from __future__ import annotations

from .. import SkillSpec
from .context import PlanContext
from .prompt import SYSTEM_TEMPLATE, render_plan_task
from .runner import parse_plan, run_planner

SKILL = SkillSpec(
    name="plan_writing",
    version="1.0.0",
    description="按固定骨架把专家意见与共识终态写成变更方案（骨架结构化、正文自由、逐条可回溯）",
    inputs="PlanContext",
    outputs="PlanDraft",
    tools=(),
)

__all__ = [
    "SKILL",
    "SYSTEM_TEMPLATE",
    "PlanContext",
    "parse_plan",
    "render_plan_task",
    "run_planner",
]
