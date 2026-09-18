"""方案撰写的执行器：渲染 → 调用 → 解析 → 契约修复重试（与 `expert_review` 同构）。

与专家评审的**两点不同**，都是刻意的：

* 失败**不降级为弃权**：方案没有「弃权」这种产物。重试耗尽就抛 ``ContractViolation``，由外环
  决定回落到确定性模板（`plan_source="template"`）——**降级必须显式**（P5）。
* 解析时**丢掉没有证据的条目**：``PlanClaim.evidence_ids`` 在契约层要求非空，那种内容本该写进
  `assumption_notes`（提示词已写明），解析器只是不配合伪造引用。
"""

from __future__ import annotations

from ....contracts import (
    PLAN_HEADING_LABELS,
    LLMCallMeta,
    PlanClaim,
    PlanDraft,
    PlanSection,
    Usage,
)
from ....errors import ContractViolation
from ....ports import EventSinkPort, LLMPort
from ..expert_review.runner import extract_json
from .context import PlanContext
from .prompt import render_plan_task

PLANNER = "planner"


def parse_plan(raw: str) -> PlanDraft:
    """把模型输出解析成 ``PlanDraft``。

    这里只管**契约级**的事（条目必须有引用、必须有章节），**不查 registry**——「引用的 id 是不是
    本轮真的检索到了」是守卫的判据（`workflow.plan_guard`），两处都做会让那条判据永远不触发。
    解析器丢的只是「压根没写引用」的条目（那种内容本该进 `assumption_notes`）。
    """
    payload = extract_json(raw)

    sections: list[PlanSection] = []
    for raw_section in payload.get("sections", []):
        if not isinstance(raw_section, dict):
            continue
        heading = str(raw_section.get("heading", ""))
        if heading not in PLAN_HEADING_LABELS:
            # 自创章节 ⇒ 让它带着反馈重试：骨架必须由契约固定（D-98），否则「必需章节有没有」
            # 这件事根本无从判断。
            raise ContractViolation(
                f"自创章节 {heading!r}（只允许 {sorted(PLAN_HEADING_LABELS)}）", node=PLANNER
            )
        claims: list[PlanClaim] = []
        for item in raw_section.get("claims", []):
            if not isinstance(item, dict) or not item.get("text"):
                continue
            cited = [str(eid) for eid in item.get("evidence_ids", []) if str(eid)]
            if not cited:
                continue
            claims.append(PlanClaim(text=str(item["text"]), evidence_ids=cited))
        sections.append(
            PlanSection(heading=heading, body=str(raw_section.get("body", "")), claims=claims)
        )

    if not sections:
        raise ContractViolation("方案没有任何章节", node=PLANNER)
    notes = [str(item) for item in payload.get("assumption_notes", []) if str(item).strip()]
    return PlanDraft(sections=sections, assumption_notes=notes)


async def run_planner(
    context: PlanContext,
    llm: LLMPort,
    *,
    max_contract_retries: int = 1,
    events: EventSinkPort | None = None,
) -> tuple[PlanDraft, Usage]:
    """撰写一版方案。**重试耗尽即抛** ``ContractViolation``（外环据此回落模板）。"""
    system, user = render_plan_task(context)
    version = (context.previous_plan.version + 1) if context.previous_plan else 1
    # 带外调用上下文（§12 1f）：方案没有「轮次」，用版本号占位，缓存键同样覆盖它。
    meta = LLMCallMeta(expert=PLANNER, round=version, mode="plan")
    usage = Usage()
    last_error = ""

    for attempt in range(1, max_contract_retries + 2):
        result = await llm.complete(purpose="plan", system=system, user=user, meta=meta)
        usage = usage + result.usage
        try:
            plan = parse_plan(result.content)
        except ContractViolation as exc:
            last_error = str(exc)
            if attempt > max_contract_retries:
                break
            user = (
                user
                + f"\n\n## 上次输出无法解析\n{last_error}\n请只输出一个合法 JSON 对象。"
            )
            continue
        plan.version = version
        return plan, usage

    raise ContractViolation(f"方案契约重试耗尽：{last_error}", node=PLANNER)
