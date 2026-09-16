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
    MemoryView,
)
from .experts import select_experts
from .guard import LoopState

UNIFORM_WEIGHT = 1.0


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
            if self._evidence_gap(state):
                proposals.append(
                    ActionProposal(
                        action="request_evidence",
                        payload={
                            "query": request,
                            "reason": "上一轮分歧源于证据不同（重叠度 < 1）",
                            "scope": "cases",
                            "effective_round": state.round_no + 1,
                        },
                        rationale="缺同类案例证据；只在下一轮生效（D-16）",
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

    @staticmethod
    def _evidence_gap(state: LoopState) -> bool:
        """上一轮归因结论是「证据不同」→ 提证据请求（Q-22：归因只记录，不改控制流）。"""
        return bool(state.attributions) and state.attributions[-1].kind == "evidence"

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
