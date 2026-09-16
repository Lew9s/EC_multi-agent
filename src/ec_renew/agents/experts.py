"""专家目录：有哪些专家、怎么按规则选出候选集（design.md §0.3 / §5.4.8）。

**评审本身不在这里** —— 它是可复用技能，实现在
``agents/skills/expert_review/``（说明见其 ``SKILL.md``）。本文件只回答「谁」：
六个专业标识、关键词先验、以及确定性的规则选人。
"""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts import EXPERT_IDS

# --------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------- #





# --------------------------------------------------------------------------- #
# Expert selection (deterministic rule table)
# --------------------------------------------------------------------------- #

MIN_SELECTED_EXPERTS = 3

DISCIPLINE_PRIORS: dict[str, tuple[str, ...]] = {
    "E01": ("结构", "肋板", "扶强", "舱壁", "分段", "开孔", "加厚", "扁钢", "甲板", "主机座"),
    "E02": ("舾装", "管系", "可达", "施工顺序", "安装空间", "脚手架"),
    "E03": ("规范", "合规", "证书", "质量", "检验", "船检", "船级社", "KR"),
    "E04": ("电气", "电缆", "配电", "照明", "接地"),
    "E05": ("轮机", "管路", "阀", "泵", "辅机"),
    "E06": ("焊接", "焊", "材料", "热输入", "探伤", "割除", "代用"),
}


def select_experts(
    request: str,
    historical_disciplines: Sequence[str] = (),
    *,
    alpha_num: bool = True,
) -> list[str]:
    """Deterministic, stably-ordered expert selection.

    Order of preference:
      1. disciplines historically involved with this component (graph-grounded);
      2. keyword priors matched in the request text;
      3. all six (the safe fallback: over-evaluate rather than miss).
    """
    picked: set[str] = {e for e in historical_disciplines if e in EXPERT_IDS}

    if alpha_num and not picked:
        for expert, keywords in DISCIPLINE_PRIORS.items():
            if any(kw in request for kw in keywords):
                picked.add(expert)

    if not picked:
        picked = set(EXPERT_IDS)

    # Structural or material changes always pull in quality review.
    if picked & {"E01", "E06"}:
        picked.add("E03")

    # Floor: a single expert cannot reach quorum, and missing a domain is worse
    # than over-evaluating -> fall back to all six (design rule §5.5).
    if len(picked) < MIN_SELECTED_EXPERTS:
        picked = set(EXPERT_IDS)

    return [e for e in EXPERT_IDS if e in picked]
