"""元智能体：决策部分（§5.4.10）—— 规则骨架（Q-06「三明治」的第一层）。

现在只有骨架：激活集 / 权重 / 证据子集全部由**确定性规则**产出（可复现），LLM 只负责
「补骨架未覆盖的差集」的那一层尚未接入（Q-06 已定方向，现状见 D-89）。

三条边界写在类型与函数里，而不是靠注释提醒：

* 载荷经 ``plan_payload()`` 产出，**去掉 rationale** —— 元智能体的推理永不进任何子智能体
  prompt（D-76），且载荷键与守卫的 ``DISPATCH_PAYLOAD_KEYS`` 闭集逐字一致（D-75）。
* ``think()`` 只产出**提案**，一律不碰 registry / 投影 / 额度（D-71）。
* 裁定（共识、停滞）不在这里 —— 由外环以回调注入（D-73）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..contracts import (
    EXPERT_IDS,
    ActionProposal,
    ActivationPlan,
    EvidenceRequest,
    MemoryView,
)
from .experts import select_experts
from .guard import LoopState

UNIFORM_WEIGHT = 1.0

#: 证据缺口的**词表 → scope** 映射（D-94）。固定词表让「缺什么 → 查什么」可复现、可评审：
#: 命中的词会被拼进 query，而模型原文（``uncertainties``）**永远不进** query。
GAP_SCOPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "standards",
        ("规范", "标准", "船级社", "船检", "证书", "认可",
         "wps", "pqr", "探伤", "ndt", "验收", "检验"),
    ),
    ("cases", ("同类", "历史", "类似", "先例", "案例", "以往")),
    (
        "components",
        ("板厚", "材质", "规格", "钢级", "图纸", "图号", "补强", "强度", "疲劳", "肋位", "开孔"),
    ),
    ("departments", ("部门", "签收", "船东", "设计公司")),
)

#: 每轮最多提几条请求、每条最多带几个关键词（防空转；Q-04 的检索配额留给检索端落地时定）。
MAX_GAP_REQUESTS = 2
MAX_GAP_KEYWORDS = 4


def plan_payload(plan: ActivationPlan) -> dict[str, object]:
    """把 ``ActivationPlan`` 转成提案载荷：**丢弃 rationale**（D-76）。

    载荷键必须与守卫的 ``DISPATCH_PAYLOAD_KEYS`` 闭集一致——多一个键就整单作废，
    这正是「分发通道零自由文本」成为结构性约束而非承诺的地方（D-75）。
    """
    return {
        "active_experts": list(plan.active_experts),
        "weights": dict(plan.weights),
        "evidence_scope": {k: list(v) for k, v in plan.evidence_scope.items()},
        "cross_domain_flags": list(plan.cross_domain_flags),
    }


def gap_payload(gap: EvidenceRequest) -> dict[str, object]:
    """把缺口转成 act 载荷：**只含 ``request_evidence`` 允许的键**（D-90）。

    ``EvidenceRequest`` 自带 ``expert`` 字段，而 act 载荷键是**闭集**——多一个键整单作废。
    守卫拒收是对的（闭集闸门正是在防「没申报的键混进管道」），所以由生产者裁掉它，
    而不是为了通过而放宽守卫。
    """
    return {
        "query": gap.query,
        "reason": gap.reason,
        "scope": gap.scope,
        "effective_round": gap.effective_round,
    }


@dataclass(frozen=True)
class RuleSkeleton:
    """确定性决策骨架（§5.4.8）。"""

    def plan(
        self,
        *,
        request: str,
        historical_disciplines: Sequence[str] = (),
        baseline_ids: Sequence[str] = (),
        active_override: Sequence[str] | None = None,
        weights_override: Mapping[str, float] | None = None,
    ) -> ActivationPlan:
        """规则出骨架：图 ``disciplines`` 命中 → 关键词先验 → 最小专家数兜底。

        ``active_override`` 是**人类在 HITL 里改过的专家集**（D-19：用户调整不受限
        但留痕）；它优先于规则表。
        """
        if active_override:
            allowed = set(active_override)
            active = [expert for expert in EXPERT_IDS if expert in allowed]
        else:
            active = select_experts(request, historical_disciplines)

        if weights_override:
            weights = {e: float(weights_override.get(e, UNIFORM_WEIGHT)) for e in active}
        else:
            # §5.4.7 要求 Σ=1：骨架**直接**产出归一化权重，守卫因此不必再修正它。
            uniform = UNIFORM_WEIGHT / len(active) if active else UNIFORM_WEIGHT
            weights = {expert: uniform for expert in active}

        return ActivationPlan(
            active_experts=active,
            weights=weights,
            evidence_scope={expert: list(baseline_ids) for expert in active},
            rationale="规则骨架：图 disciplines + 关键词先验 + 最小专家数（Q-06 第一层）",
        )

    def think(
        self,
        state: LoopState,
        view: MemoryView,
        *,
        request: str,
        historical_disciplines: Sequence[str] = (),
        active_override: Sequence[str] | None = None,
        weights_override: Mapping[str, float] | None = None,
    ) -> list[ActionProposal]:
        """一轮的 act 序列（确定性）。``view`` 是常驻层给的证据基线与计数。"""
        proposals: list[ActionProposal] = []

        if state.round_no >= 2:
            proposals.append(
                ActionProposal(
                    action="attribute", payload={}, rationale="先归因上一轮分歧（§5.2.6）"
                )
            )
            # L4 不常驻（D-74）：轮次折叠明细按需经 read_memory 取，并计入每轮额度。
            proposals.append(
                ActionProposal(
                    action="read_memory",
                    payload={"view": "rounds"},
                    rationale="取 L4 轮次折叠明细",
                )
            )

        plan = self.plan(
            request=request,
            historical_disciplines=historical_disciplines,
            baseline_ids=view.evidence_ids,
            active_override=active_override,
            weights_override=weights_override,
        )
        proposals.append(
            ActionProposal(
                action="dispatch_experts",
                payload=plan_payload(plan),
                rationale=plan.rationale,
            )
        )
        return proposals

    def gaps(self, state: LoopState, *, request: str) -> list[EvidenceRequest]:
        """由专家**已经写好的待补清单**派生证据请求（D-94：确定性、词表驱动）。

        **不用「归因结论 = 证据不同」当触发信号**：全员弃权时根本没有分歧可归因，那条路不可达。
        而「缺什么」就写在专家的 ``uncertainties`` 与 ``constraints`` 里，是**最可执行**的信号；
        只让它沉在渲染文本里，机器消费者就拿不到。

        两条纪律：

        * ``query`` 由**固定词表 + 原始请求**拼成，**绝不把 ``uncertainties`` 原文当检索输入**：
          那等于让模型文本驱动检索 → 检索结果进下一轮事实基线，与 D-71「概率组件不直接写事实」
          同一精神；用词表拼接则检索行为**可复现**。
        * 只在**已知缺口**上提请求：命中词为空就不提，宁缺勿滥。
        """
        text = " ".join(
            [item for op in state.latest.values() for item in op.uncertainties]
            + [item for op in state.latest.values() for item in op.constraints]
        ).lower()
        if not text:
            return []

        requests: list[EvidenceRequest] = []
        for scope, words in GAP_SCOPE_KEYWORDS:
            hits = [word for word in words if word in text][:MAX_GAP_KEYWORDS]
            if not hits:
                continue
            requests.append(
                EvidenceRequest(
                    query=f"{request}｜需补充：{'、'.join(hits)}",
                    reason=f"专家在 uncertainties/constraints 中声明缺少：{'、'.join(hits)}",
                    scope=scope,  # type: ignore[arg-type]
                    effective_round=state.round_no + 1,
                )
            )
            if len(requests) >= MAX_GAP_REQUESTS:
                break
        return requests

    def closing(self, approved: bool, stalled: bool) -> ActionProposal:
        """收束提案（§5.4.1）：达标/停滞 → finalize；未达共识 → ask_human。"""
        if approved or stalled:
            return ActionProposal(
                action="finalize", payload={}, rationale="可收束：证据支撑与保证等级成立"
            )
        return ActionProposal(
            action="ask_human",
            payload={"reason": "consensus_not_reached"},
            rationale="未达共识 → 交人工；何时真正挂起由外环决定（D-73）",
        )
