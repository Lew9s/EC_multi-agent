"""``plan_writing`` 的输入契约（D-98）。

放在技能包里而不是 `contracts.py`：它是**这个技能的输入形状**，由外环（`workflow.build_plan_context`）
确定性投影出来，不是跨模块的公共契约。`ExpertTask` 放在 `contracts.py` 是因为它是内环派发通道的
载荷；本结构只走外环 → 技能这一条边。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ....contracts import ExpertOpinion, PlanDraft


@dataclass(frozen=True)
class PlanContext:
    """方案撰写者能看到的一切。**只有结构化字段，没有自由文本通道**（除人类事实原文）。"""

    request: str
    #: **完整**意见，含专家身份（D-100：撰写者是汇总者，需要身份做专业归口）。
    opinions: tuple[ExpertOpinion, ...] = ()
    #: 本轮冻结基线的证据：(evidence_id, gist)。这是模型**唯一**可引用的证据集合。
    evidence: tuple[tuple[str, str], ...] = ()
    consensus_status: str = ""
    review_reason: str | None = None
    #: 必须全部落进方案的前置条件（D-95）。
    conditions: tuple[str, ...] = ()
    grounding_basis: str = ""
    assurance_level: str = ""
    #: 人类提供的事实（原文保留，D-28）。
    human_facts: tuple[str, ...] = ()
    #: 会话上下文：历轮摘要 + 用户约束（§5.9）。
    session_notes: tuple[str, ...] = ()
    #: 迭代回路：上一版方案 + 用户逐条意见（D-99）。
    previous_plan: PlanDraft | None = None
    plan_feedback: tuple[str, ...] = field(default=())
