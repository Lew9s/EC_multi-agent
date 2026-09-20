"""元智能体：决策部分（§5.4.10）—— 规则骨架 + LLM 补差集（Q-06 的「三明治」）。

两层都在这里：``RuleSkeleton`` 是**第一层**（确定性、可复现），``run_decision`` 是**第二层**
（LLM 只补骨架未覆盖的差集）。合并、校验、归一化由 ``merge_decision`` 这段代码负责，结果照旧
要过守卫——**没有任何一层能直接写 registry / 投影 / 额度**（D-71）。

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
    LLMCallMeta,
    MemoryView,
    MetaDecision,
    Usage,
)
from ..errors import ContractViolation
from ..ports import EventSinkPort, LLMPort
from .experts import select_experts
from .guard import LoopState
from .skills.expert_review.runner import extract_json

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
        plan_override: ActivationPlan | None = None,
    ) -> list[ActionProposal]:
        """一轮的 act 序列（确定性）。``view`` 是常驻层给的证据基线与计数。

        ``plan_override`` 是决策层合并后的首轮激活方案（D-102）：只作用于**第 1 轮**。
        第 2 轮起回到 ``plan()``，因为证据子集要跟着**当前**冻结基线走，而决策层只跑一次、
        它的子集是按首轮基线裁的——沿用一份过期裁剪，等于让专家在第 2 轮看不到新证据。
        """
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

        plan = (
            plan_override
            if plan_override is not None and state.round_no == 1
            else self.plan(
                request=request,
                historical_disciplines=historical_disciplines,
                baseline_ids=view.evidence_ids,
                active_override=active_override,
                weights_override=weights_override,
            )
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


# --------------------------------------------------------------------------- #
# 三明治第二层：LLM 只补差集（Q-06 / §5.4.8 / D-102）
# --------------------------------------------------------------------------- #

#: 决策层的节点名：StepRecord 与事件日志用它，错误信息也用它。
DECISION_NODE = "meta_decision"

#: 渲染进决策提示词的证据上限。L3 层本来就是一具有界视图（§5.4.3），决策层只需要
#: 「有哪些证据、各讲什么」，不需要整个 registry。
MAX_DECISION_EVIDENCE = 24

DECISION_SYSTEM = """你是工程变更评审的**决策助手**，服务对象是元智能体。

已经有一份**确定性规则骨架**（图 disciplines + 关键词先验 + 最小专家数）给出了激活集与权重。
你的唯一任务是**补差集**：规则没覆盖到、但这次变更确实需要的专业，以及每条证据该给哪个专业看。

硬规则（违反会被程序拒绝；越界即整份作废并回落到纯骨架）：

1. **只能加，不能删**。骨架已激活的专家一律保留，试图移除会被程序加回。
2. `add_experts` 只能取这些 id：{experts}，且不要重复骨架已激活的。
3. `weights` 可选：键只能是激活专家，值为非负数；省略即沿用骨架权重。
4. `evidence_scope` 可选：键只能是激活专家，值只能引用「本轮可引用证据」里出现过的 id。
   **不确定就给空**——程序会把空子集回落成整份基线；凭印象少给证据，只会让判断变差。
5. `rationale` 写清「为什么补这些」。它**不会**被任何子智能体看到（D-76）。
6. 只输出一个 JSON 对象，不要 markdown 代码围栏。

输出格式：
{{"add_experts": ["E0x"], "weights": {{"E01": 1.5}}, "evidence_scope": {{"E01": ["E-…"]}}, "rationale": "……"}}
"""


@dataclass(frozen=True)
class MetaDecisionContext:
    """决策层能看到的一切（§5.4.3 的 L1/L2/L3 三层）。

    **会话层只到这里**（§8.4.5：元智能体是会话历史的唯一读者）。它进的是决策提示词，而决策的
    输出是闭集 id（激活集 / 权重 / 证据 id）——因此会话内容**不可能**经派发通道流进子智能体
    的 prompt。
    """

    request: str
    skeleton: ActivationPlan
    historical_disciplines: tuple[str, ...] = ()
    #: 本轮可引用证据：(evidence_id, gist)。它就是 ``evidence_scope`` 的合法取值域。
    evidence: tuple[tuple[str, str], ...] = ()
    #: L1 会话层：锚点请求、最近几轮结论摘要、用户明确约束。
    anchor_request: str = ""
    recent_conclusions: tuple[str, ...] = ()
    user_constraints: tuple[str, ...] = ()


def render_decision_task(context: MetaDecisionContext) -> tuple[str, str]:
    """渲染一次决策任务，返回 ``(system, user)``。

    **确定性**：同样的输入渲染出同样的文本——决策也要可复现，否则同一条请求两次跑出两个
    激活集，P7 与「同输入同基线」立刻不成立。
    """
    system = DECISION_SYSTEM.format(experts="、".join(EXPERT_IDS))
    active = "、".join(context.skeleton.active_experts) or "（无）"
    optional = [e for e in EXPERT_IDS if e not in set(context.skeleton.active_experts)]
    parts = [
        f"## 变更请求\n{context.request}",
        f"## 规则骨架已激活（不能删）\n{active}",
        f"## 可补的专家\n{'、'.join(optional) or '（无）'}",
    ]
    if context.historical_disciplines:
        parts.append("## 历史同类变更涉及的专业\n" + "、".join(context.historical_disciplines))

    if context.evidence:
        lines = "\n".join(f"- {eid}: {gist}" for eid, gist in context.evidence)
        parts.append(f"## 本轮可引用证据（evidence_scope 只能引用下列 id）\n{lines}")
    else:  # pragma: no cover - 基线永远含请求本身
        parts.append("## 本轮可引用证据\n（无）")

    session: list[str] = []
    if context.anchor_request and context.anchor_request != context.request:
        session.append(f"- 会话锚点（首轮请求）：{context.anchor_request}")
    session.extend(f"- {note}" for note in context.recent_conclusions)
    session.extend(f"- 用户约束：{item}" for item in context.user_constraints)
    if session:
        parts.append("## 会话上下文（只有你能看到）\n" + "\n".join(session))

    return system, "\n\n".join(parts)


def parse_decision(raw: str, *, allowed_evidence: Sequence[str]) -> MetaDecision:
    """解析决策输出，并在**契约层**挡掉越界。

    两条边界是硬性的（§5.4「元智能体不得编造 evidence_id」）：

    * ``add_experts`` / 权重 / 证据子集的键只能是六个专家 id；
    * 证据子集的值只能来自**渲染进提示词的那批 id**。

    越界一律抛 ``ContractViolation`` → 带着反馈重试 → 重试耗尽则整层降级。这里**不做**
    静默丢弃：静默丢弃会让「模型编了个 id」这件事在产物里看不出来。
    """
    payload = extract_json(raw)
    known_evidence = set(allowed_evidence)

    add: list[str] = []
    for item in payload.get("add_experts") or []:
        expert = str(item)
        if expert not in EXPERT_IDS:
            raise ContractViolation(f"补差集出现未知专家 id {expert!r}", node=DECISION_NODE)
        if expert not in add:
            add.append(expert)

    weights: dict[str, float] = {}
    raw_weights = payload.get("weights") or {}
    if isinstance(raw_weights, Mapping):
        for key, value in raw_weights.items():
            expert = str(key)
            if expert not in EXPERT_IDS:
                raise ContractViolation(f"权重给了未知专家 {expert!r}", node=DECISION_NODE)
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ContractViolation(
                    f"权重不是数值：{expert}={value!r}", node=DECISION_NODE
                ) from exc
            if number < 0:
                raise ContractViolation(f"权重为负：{expert}={number}", node=DECISION_NODE)
            weights[expert] = number

    scope: dict[str, list[str]] = {}
    raw_scope = payload.get("evidence_scope") or {}
    if isinstance(raw_scope, Mapping):
        for key, ids in raw_scope.items():
            expert = str(key)
            if expert not in EXPERT_IDS:
                raise ContractViolation(f"证据子集给了未知专家 {expert!r}", node=DECISION_NODE)
            kept = sorted({str(i) for i in ids})  # type: ignore[union-attr]
            unknown = [i for i in kept if i not in known_evidence]
            if unknown:
                raise ContractViolation(
                    f"{expert} 的证据子集含本轮不可引用的 id：{unknown[:3]}", node=DECISION_NODE
                )
            if kept:
                scope[expert] = kept

    return MetaDecision(
        add_experts=add,
        weights=weights,
        evidence_scope=scope,
        rationale=str(payload.get("rationale", "")),
    )


async def run_decision(
    context: MetaDecisionContext,
    llm: LLMPort,
    *,
    max_contract_retries: int = 1,
    events: EventSinkPort | None = None,
) -> tuple[MetaDecision, Usage]:
    """让 LLM 补一次差集。

    **重试耗尽即抛** ``ContractViolation``：这一层没有「部分决策」可言，降级（回落到纯骨架）
    是调用方的事——runtime 把它变成值，外环落事件。异常语义与 ``plan_writing`` 一致。
    """
    system, user = render_decision_task(context)
    # 带外调用上下文（§12 1f）：决策没有「轮次」，固定用第 1 轮——它只在首轮跑一次。
    meta = LLMCallMeta(expert=DECISION_NODE, round=1, mode="decision")
    allowed_evidence = [eid for eid, _ in context.evidence]
    usage = Usage()
    last_error = ""

    for attempt in range(1, max_contract_retries + 2):
        result = await llm.complete(
            purpose="meta_decision", system=system, user=user, meta=meta
        )
        usage = usage + result.usage
        try:
            decision = parse_decision(result.content, allowed_evidence=allowed_evidence)
        except ContractViolation as exc:
            last_error = str(exc)
            if events is not None:
                events.emit(
                    "contract_violation", node=DECISION_NODE, attempt=attempt, error=last_error
                )
            if attempt > max_contract_retries:
                break
            user = user + f"\n\n## 上次输出无法解析\n{last_error}\n请只输出一个合法 JSON 对象。"
            continue
        return decision, usage

    raise ContractViolation(f"决策契约重试耗尽：{last_error}", node=DECISION_NODE)


def merge_decision(
    skeleton: ActivationPlan,
    decision: MetaDecision | None,
    *,
    baseline_ids: Sequence[str],
) -> tuple[ActivationPlan, tuple[str, ...]]:
    """把 LLM 的差集并进规则骨架（Q-06 的「代码负责合并、校验、归一化」那一层）。

    ``decision is None``（LLM 不可用 / 契约越界）时**原样返回骨架**，corrections 为空——
    降级本身由调用方落事件，这里再表达一次只会让同一件事出现两遍。

    合并规则：

    * **只增不减**：骨架已激活的专家一律保留（守卫还有一道同样的判据；这里做是为了让
      corrections 记的是「模型越界」而不是「程序没合并好」）。
    * 权重按骨架的相对比例作底，LLM 只覆盖它明确给了值的专家，然后归一化到 Σ=1（§5.4.7）
      ——归一化在这里做完，守卫就不必每轮再修正一次。
    * 证据子集与基线求交（D-25）；给了空集或全部越界的专家回到整份基线（§4 不变量 4：
      无证据就不给该专家派发事实）。
    """
    if decision is None:
        return skeleton, ()

    corrections: list[str] = []
    skeleton_active = set(skeleton.active_experts)
    active = [
        e for e in EXPERT_IDS if e in skeleton_active | set(decision.add_experts)
    ]
    added = [e for e in active if e not in skeleton_active]
    if added:
        corrections.append(f"补差集：{'、'.join(added)}")

    raw = {e: float(skeleton.weights.get(e, 1.0)) for e in active}
    for expert, value in decision.weights.items():
        if expert in raw:
            raw[expert] = max(0.0, float(value))
        else:
            corrections.append(f"忽略非激活专家的权重：{expert}")
    total = sum(raw.values())
    if total <= 0:  # pragma: no cover - 骨架权重恒为正，走到这里说明上游坏了
        raw = {e: 1.0 for e in active}
        total = float(len(active)) or 1.0
    weights = {e: raw[e] / total for e in active}

    baseline = set(baseline_ids)
    scope: dict[str, list[str]] = {}
    for expert in active:
        wanted = decision.evidence_scope.get(expert)
        if not wanted:
            scope[expert] = list(baseline_ids)
            continue
        kept = sorted(set(wanted) & baseline)
        dropped = len(set(wanted)) - len(kept)
        if dropped:
            corrections.append(f"{expert} 的证据子集剔除了 {dropped} 条越界 id（D-25）")
        # 裁剪后为空 ⇒ 回落整份基线：给一个专家「零证据」不是裁剪，是让他没法判断。
        scope[expert] = kept or list(baseline_ids)

    rationale = skeleton.rationale
    if decision.rationale:
        rationale = f"{rationale}；LLM 补差集：{decision.rationale}"
    return (
        ActivationPlan(
            active_experts=active,
            weights=weights,
            evidence_scope=scope,
            rationale=rationale,
        ),
        tuple(corrections),
    )
