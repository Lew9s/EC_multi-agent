"""方案撰写的提示词（D-98）。

三条纪律写进提示词，且都有守卫兜底（提示词不是唯一防线，§3.1）：

1. **骨架固定**：章节只能是给定的那几个 key，其中四个必须给出；
2. **每条 claim 必须引用「本轮可引用证据」里出现过的 `evidence_id`** —— 写不出证据的话，
   放进 `assumption_notes`，不要伪造引用；
3. **前置条件必须全部出现在方案里**（D-95：少一条，「有条件通过」就变成「无条件通过」）。
"""

from __future__ import annotations

from ....contracts import PLAN_HEADING_LABELS, REQUIRED_PLAN_HEADINGS
from .context import PlanContext

SYSTEM_TEMPLATE = """你是工程变更方案撰写人。你的读者是施工、采购与质控人员。

你的任务：把已经完成的**专家评审**写成一份可以直接执行的**变更方案**。

硬规则（违反会被程序拒绝）：

1. 章节只能用这些 key，不要自创：
{headings}
   其中**必给**（缺一即不合格）：{required}
2. 每一条 `claims[].text` 都必须引用 `evidence_ids`，且只能引用「本轮可引用证据」里出现过的 id。
   全篇不得出现没有证据支撑的事实性条目。
3. 写不出证据的内容（假设、常规做法、待确认项）**只能**放进 `assumption_notes`。
4. 下列前置条件**必须全部**出现在方案里（原文或等价表述）：
{conditions}
5. 不要重复评审结论的措辞，要写**怎么做**：范围、依据、施工与采购要求、风险与前置条件。
6. 只输出一个 JSON 对象，不要 markdown 代码围栏。

输出格式：
{{"sections": [{{"heading": "scope", "body": "……", "claims": [{{"text": "……", "evidence_ids": ["E-…"]}}]}}],
  "assumption_notes": ["……"]}}
"""


def _headings_block() -> str:
    return "\n".join(f"   - {key}（{label}）" for key, label in PLAN_HEADING_LABELS.items())


def render_plan_task(context: PlanContext) -> tuple[str, str]:
    """把一次收束的评审渲染成方案撰写任务，返回 ``(system, user)``。

    **确定性**：同样的输入渲染出同样的文本（可复现是 P7 的硬要求，也是「同样的评审不会写出
    两个不同的方案骨架」这一验收条件的前提）。
    """
    required = "、".join(sorted(REQUIRED_PLAN_HEADINGS))
    conditions = "\n".join(f"   - {item}" for item in context.conditions) or "   （无）"
    system = SYSTEM_TEMPLATE.format(
        headings=_headings_block(), required=required, conditions=conditions
    )

    parts: list[str] = [f"## 变更请求\n{context.request}"]

    if context.evidence:
        lines = "\n".join(f"- {eid}: {gist}" for eid, gist in context.evidence)
        parts.append(f"## 本轮可引用证据（claims 只能引用下列 id）\n{lines}")
    else:  # pragma: no cover - 基线永远含请求本身，走不到
        parts.append("## 本轮可引用证据\n（无）")

    parts.append(f"## 共识终态\n状态：{context.consensus_status}"
                 + (f"｜交人原因：{context.review_reason}" if context.review_reason else "")
                 + f"\n依据等级：{context.grounding_basis}｜保证等级：{context.assurance_level}")

    if context.opinions:
        # 完整意见 + 身份：撰写者要做专业归口（D-100）。这里**不做匿名化**，是有意的。
        blocks: list[str] = []
        for opinion in sorted(context.opinions, key=lambda op: op.expert):
            lines = [
                (
                    f"### {opinion.expert}（决策：{opinion.decision}｜风险：{opinion.risk_level}｜"
                    f"依据类型：{opinion.basis}）"
                ),
                f"理由：{opinion.rationale or '（无）'}",
            ]
            if opinion.constraints:
                lines.append("约束：" + "；".join(opinion.constraints))
            if opinion.uncertainties:
                lines.append("不确定：" + "；".join(opinion.uncertainties))
            lines.append("引用证据：" + ", ".join(sorted(opinion.evidence_ids)))
            blocks.append("\n".join(lines))
        parts.append("## 专家意见（含身份，供你做专业归口）\n" + "\n\n".join(blocks))

    if context.human_facts:
        parts.append("## 用户提供的事实（原文，视为一等证据）\n" + "\n".join(f"- {f}" for f in context.human_facts))

    if context.previous_plan is not None:
        parts.append(
            "## 上一版方案（本轮是**迭代**，不是重写）\n"
            + _render_plan_plain(context.previous_plan)
        )
    if context.plan_feedback:
        parts.append(
            "## 用户对本方案的意见（逐条落实；与证据冲突时在 assumption_notes 里说明）\n"
            + "\n".join(f"- {item}" for item in context.plan_feedback)
        )

    if context.session_notes:
        parts.append("## 会话上下文\n" + "\n".join(f"- {item}" for item in context.session_notes))

    return system, "\n\n".join(parts)


def _render_plan_plain(plan: object) -> str:
    """把上一版方案渲染成纯文本（供迭代时对照）。"""
    lines: list[str] = []
    for section in getattr(plan, "sections", []):
        label = PLAN_HEADING_LABELS.get(section.heading, section.heading)
        lines.append(f"### {label}")
        if section.body:
            lines.append(section.body)
        for claim in section.claims:
            lines.append(f"- {claim.text}（依据 {', '.join(claim.evidence_ids)}）")
    return "\n".join(lines)
