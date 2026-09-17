"""The outer loop.

A deterministic workflow that *hosts* agents (docs/design.md §4). Every stage is
a plain function so it can be tested on its own. Nothing here is LLM-decided
except the free-text content produced by the experts themselves.

Invariants enforced in this module
----------------------------------
1. One frozen baseline per round; every expert in a round sees the same one.
2. Parallel results are merged in **sorted** order, so a live run and a replay
   produce byte-identical output.
3. Consensus denominator is the **configured** weight sum, so an abstaining
   expert cannot inflate the score (fixes the defect found in CDIACR).
4. No evidence -> no expert dispatch (the no-history branch).
5. An expert never sees another expert's judgment or the session history.
6. A round that adds no evidence and moves no prompt-affecting field is a fixed
   point: it is reported as ``stalled`` instead of being silently bounded by
   ``max_rounds`` (D-81/D-82).
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from functools import partial

from .agents.experts import select_experts
from .agents.guard import Guard
from .agents.memory import MemoryService, collect_hard_constraints
from .agents.meta import RuleSkeleton
from .agents.runtime import META_SPEC, AgentRuntime
from .config import settings as default_settings
from .contracts import (
    DECISION_SCORES,
    DISCIPLINE_NAMES,
    EXPERT_IDS,
    AssuranceLevel,
    EntityRef,
    EvidenceBundle,
    EvidenceMeta,
    EvidenceRequest,
    ExpertOpinion,
    GraphExpansion,
    Grounding,
    HumanReviewRequest,
    IntentCompletion,
    ReviewReason,
    RunInput,
    RunResult,
    SubQuestion,
)
from .ports import RunContext as _RunContext

MAX_QUERIES = 5


# --------------------------------------------------------------------------- #
# Stage 1 — intent completion
# --------------------------------------------------------------------------- #


def build_queries(request: str, entities: Sequence[EntityRef]) -> list[str]:
    """Multi-route probe queries, deterministic and de-duplicated."""
    queries = [request.strip()]
    for entity in entities:
        queries.append(entity.name)
        queries.append(f"{entity.name} 历史变更")
    seen: list[str] = []
    for query in queries:
        query = query.strip()
        if query and query not in seen:
            seen.append(query)
    return seen[:MAX_QUERIES]


async def complete_intent(
    run_input: RunInput, ctx: _RunContext
) -> tuple[IntentCompletion, list[EntityRef], GraphExpansion]:
    request = run_input.request.strip()
    entities: list[EntityRef] = []
    expansion = GraphExpansion()
    warnings: list[str] = []

    if ctx.retriever is not None:
        entities = await ctx.retriever.link(request)
        expansion = await ctx.retriever.expand(entities)
    if not entities:
        warnings.append("no_entity_linked")

    sub_questions: list[SubQuestion] = []
    query_set = build_queries(request, entities)

    intent = IntentCompletion(
        raw_request=request,
        normalized_request=" ".join(request.split()),
        mentioned_entities=entities,
        graph_expansion=expansion,
        sub_questions=sub_questions,
        query_set=query_set,
        provenance="graph" if entities else "rule",
    )
    ctx.events.emit(
        "intent_completed",
        entities=[e.name for e in entities],
        disciplines=expansion.historical_disciplines,
        queries=query_set,
        warnings=warnings,
    )
    return intent, entities, expansion


# --------------------------------------------------------------------------- #
# Stage 2 — baseline retrieval + freeze
# --------------------------------------------------------------------------- #


async def prefetch_baseline(
    intent: IntentCompletion, ctx: _RunContext, top_k: int
) -> EvidenceBundle:
    # The request itself is a fact: registering it guarantees a non-empty
    # baseline and keeps the "evidence_ids non-empty" rule satisfiable.
    ctx.registry.register(
        source="text",
        content=f"变更请求：{intent.normalized_request}",
        entity_keys=[e.name for e in intent.mentioned_entities],
        disciplines=intent.historical_disciplines,
    )

    bundle = EvidenceBundle(round=0)
    if ctx.retriever is not None:
        bundle = await ctx.retriever.prefetch(intent.query_set, top_k)
        # The registry owns ids (content-addressed). Retriever-side ids are
        # only labels, so remap them here to keep one source of truth.
        remapped: list[EvidenceMeta] = []
        for item in bundle.items:
            registered = ctx.registry.register(
                source=item.source,
                content=item.content or item.evidence_id,
                entity_keys=item.entity_keys,
                disciplines=item.disciplines,
                group_keys=item.group_keys,
                score=item.score,
            )
            remapped.append(item.model_copy(update={"evidence_id": registered.evidence_id}))
        bundle = bundle.model_copy(update={"items": remapped})
    bundle.baseline_ids = ctx.registry.all_ids()
    ctx.events.emit(
        "baseline_frozen",
        count=len(bundle.baseline_ids),
        warnings=list(bundle.warnings),
    )
    return bundle


def assess_grounding(bundle: EvidenceBundle) -> Grounding:
    """Binary branch: real historical cases, or none (docs/design.md §5.5.1)."""
    graph_items = [item for item in bundle.items if item.source == "graph"]
    if graph_items:
        return Grounding(
            basis="history",
            case_ids=[item.evidence_id for item in graph_items],
            note=f"命中 {len(graph_items)} 条历史证据",
        )
    return Grounding(
        basis="knowledge",
        case_ids=[],
        note="未命中历史案例，仅基于规范与通用工程知识",
    )


# --------------------------------------------------------------------------- #
# Stage 3 — expert dispatch
# --------------------------------------------------------------------------- #
# 派发本身（含「异常转状态」的那一个宽泛捕获点）已移入
# ``agents/runtime.AgentRuntime``：§3.1 要求它是**唯一执行入口**，不得有第二条执行路径。


# --------------------------------------------------------------------------- #
# Stage 4 — consensus
# --------------------------------------------------------------------------- #


def consensus(
    opinions: dict[str, ExpertOpinion],
    weights: dict[str, float],
    threshold: float,
    min_effective: int,
    *,
    final: bool = False,
) -> tuple[float, int, str]:
    """共识判定（D-38 / D-93 / D-95 / **D-96**）。分两层，**不要混用**：

    1. **是否收敛**（决定要不要继续迭代）——结构化规则，不是分数：
       `无 reject、且至少一位 approve、且支持度 ≥ threshold` → `approved`。
    2. **停止时如何分类**（决定交付还是交人）——只有两个出口：`approved`（交付）与
       `manual_review`（交人）。「因为什么交人」**不是状态取值的事**，而是
       :func:`manual_review_reason` 的事（D-96）。

    为什么把分数从**判据**降为**描述性支持度**（D-95）：`score = 0.5 + (a − j − x)/2` 里
    `revise` 被完全消掉，于是「全员一致附条件」与「赞反对峙」算出**同一个 0.5000**——一个无法
    区分这两种局面的数字不该拥有裁定权。分数照旧计算并报告（它仍表达「明确支持的强度」，用于
    区分「明确通过」与「交人」），但它不再决定「交给人还是交付」。

    另两条不变：分母是**配置权重和**（D-38，弃权计 0 分故只能压低分数）；`final=False` 时
    未收敛一律返回 `retry`（循环继续），终态由外环在轮次结束时以 `final=True` 取得。
    """
    # 「有效专家」= **交付了意见**的专家（D-93）。弃权是一次交付（它是一条完整的、带证据的
    # 判断），只有**执行失败**才是缺席。D-37 的原意是「避免缺席被当作通过」，此前却把「缺席」
    # 等同于了 `abstain` 这个值——于是「四位专家一致说证据不足」被读成「四位专家缺席」。
    delivered = {
        expert: op for expert, op in opinions.items() if op.abstain_kind != "execution_failure"
    }
    effective = len(delivered)
    if effective < min_effective:
        return 0.0, effective, "manual_review"

    # 全员判断性弃权不是「专家分歧未解决」，而是**证据缺口**：补救动作是补证据，不是再投票
    # （D-93）。它是**当轮收束**（即使 `final=False` 也返回非 `retry`）：它不满足「迭代能带来
    # 新信息」的前提，继续迭代只会产生共识幻觉（D-29 / §5.5.5），缺口清单在
    # `evidence_requests` 里交出。收束后的原因标签由 `manual_review_reason` 判为 `evidence_gap`。
    if delivered and all(op.decision == "abstain" for op in delivered.values()):
        return 0.0, effective, "manual_review"

    numerator = sum(
        weights.get(expert, 0.0) * DECISION_SCORES[op.decision]
        for expert, op in opinions.items()
    )
    denominator = sum(weights.get(expert, 0.0) for expert in weights)
    score = round(numerator / denominator, 4) if denominator else 0.0

    # --- 第 1 层：是否收敛（决定要不要继续迭代） -------------------------- #
    has_reject = any(op.decision == "reject" for op in delivered.values())
    has_approve = any(op.decision == "approve" for op in delivered.values())
    if not has_reject and has_approve and score >= threshold:
        return score, effective, "approved"
    if not final:
        return score, effective, "retry"

    # --- 第 2 层：停止时的分类（决定交给人还是交付） ---------------------- #
    # 唯一的另一个出口就是交人；**「为什么」由 reason 回答，不在这里分叉**（D-96）。曾经这里按
    # 「有 reject / 无 reject」分成 `manual_review` 与 `conditional`，但那个区分没有对应的行为差
    # ——收束 act 一直是 `closing(status=="approved", status=="stalled")`，`conditional` 从未因此
    # 自动交付，它实际也在问人。状态值与实际行为不一致，且多出一个消费者必须分支的取值。
    return score, effective, "manual_review"


def manual_review_reason(
    opinions: dict[str, ExpertOpinion], min_effective: int
) -> ReviewReason:
    """`manual_review` 的**原因**（D-96）——「交给人时该说的那句话」，机器可读。

    判定顺序与 :func:`consensus` 的收束顺序**逐一对应**，所以每个 `manual_review` 恰好得到一条
    原因：缺席（有效专家不足）→ 证据缺口（全员判断性弃权）→ 分歧（有人明确反对）→ 其余
    （无人反对、但支持度不足以自动交付，多为全员附条件）。

    为什么「缺席」也值得一个标签：D-37 只规定了「缺席不得被当作通过」，从未给它一个状态取值，
    于是「有人没交出意见」与「专家分歧未决」在契约里长得一模一样。既然这个字段是交人记录的一
    部分，就不能对它撒谎——把缺席说成 `conditions_only`（读起来像「条件都谈妥了」）会误导接手
    的人。这是把状态值并掉之后**新增**的一条信息，不是搬走的那两条。
    """
    delivered = {
        expert: op for expert, op in opinions.items() if op.abstain_kind != "execution_failure"
    }
    if len(delivered) < min_effective:
        return "quorum"
    if delivered and all(op.decision == "abstain" for op in delivered.values()):
        return "evidence_gap"
    if any(op.decision == "reject" for op in delivered.values()):
        return "disagreement"
    return "conditions_only"


#: 原因标签 → 交给人的那句话（报告用）。放在这里而不是渲染函数里，是为了让「原因」这件事只有
#: 一个定义处：契约的取值、判定的函数、报告的措辞三者一一对应。
REVIEW_REASON_LABELS: dict[ReviewReason, str] = {
    "quorum": "交付意见不足（有专家未交出意见）——不是分歧，是缺席，需查清原因后重跑或补评审",
    "disagreement": "存在明确反对（reject）——分歧未决，需人工裁定",
    "conditions_only": "无人明确反对，但支持度不足以自动交付——附条件方案需人工确认",
    "evidence_gap": "专家一致判断证据不足——需先补齐证据再评审",
}


def _knowledge_share(opinions: dict[str, ExpertOpinion], weights: dict[str, float]) -> float:
    """声明 ``basis="knowledge"`` 的权重占比（与共识分同口径：分母是配置权重和）。

    用来决定保证等级是否该降级：证据命中了历史案例，不代表专家用了它作依据。
    """
    total = sum(weights.get(expert, 0.0) for expert in weights)
    if not total:
        return 0.0
    knowledge = sum(
        weights.get(expert, 0.0) for expert, op in opinions.items() if op.basis == "knowledge"
    )
    return knowledge / total


# --------------------------------------------------------------------------- #
# Stage 4b — fixed-point detection
# --------------------------------------------------------------------------- #


def projected_fingerprint(opinion: ExpertOpinion) -> tuple:
    """Exactly the fields that reach the *next* round's prompt (§5.4.4, D-81).

    Included:

    * ``decision`` — drives the consensus score and the disclosed dissent count;
    * ``evidence_ids`` and ``rationale`` — come back to their own author through
      ``RevisionContext.own_previous``;
    * ``claims`` by ``claim_id`` — these reach *peers* through ``CrossAgentInfo``,
      and comparing ids rather than text makes the check immune to whitespace
      noise (D-78);
    * ``constraints`` — these reach every peer unconditionally (D-56).

    Excluded on purpose: ``uncertainties`` / ``risk_level`` / ``confidence`` /
    ``assurance`` / ``partial``. Nothing renders them into a prompt, so a change
    in them cannot alter any later round — only the final report, which is built
    from the opinions already in hand.
    """
    return (
        opinion.decision,
        tuple(sorted(opinion.evidence_ids)),
        tuple(sorted(claim.claim_id for claim in opinion.claims)),
        tuple(sorted(opinion.constraints)),
        opinion.rationale,
    )


def detect_stall(
    previous: dict[str, ExpertOpinion],
    current: dict[str, ExpertOpinion],
    previous_baseline: Sequence[str],
    current_baseline: Sequence[str],
) -> bool:
    """True when the next round could only repeat this one (§5.4.4, D-81).

    Both conditions are necessary:

    * **no new evidence** — the frozen baseline is unchanged, so every expert
      sees the same facts;
    * **no change in any prompt-affecting field** — so the only difference left
      in the next prompt is the round counter in its CTX header.

    This is why the check needs no threshold, unlike oscillation detection, and
    can therefore act immediately. The honest limit: because that counter *is*
    in the prompt, this establishes "no further information gain", not a
    byte-level proof of identical output — see docs/design.md §5.4.4.

    The alternative it replaces is worse either way: continuing yields the same
    opinions (a loop that only ``max_rounds`` stops), or yields different ones
    *because of a semantically empty counter* — which would mean enshrining
    sensitivity to an irrelevant number.
    """
    if set(current_baseline) - set(previous_baseline):
        return False
    if set(previous) != set(current):
        return False
    return all(
        projected_fingerprint(previous[expert]) == projected_fingerprint(current[expert])
        for expert in current
    )


# --------------------------------------------------------------------------- #
# Oscillation detection — recorded, never acted upon (D-81)
# --------------------------------------------------------------------------- #

#: Severity axis used *only* to decide whether a decision reversed direction.
#:
#: Deliberately not ``contracts.DECISION_SCORES``: that map exists for scoring
#: consensus, where ``abstain`` is worth the same as ``reject`` (0.0) because it
#: contributes nothing. Reusing it here would make "approve -> abstain ->
#: approve" look like a reversal — but an abstention is the *absence* of a
#: judgment, not a change of one.
_DECISION_RANK: dict[str, int] = {"reject": 0, "revise": 1, "approve": 2}


def detect_oscillations(
    history: Mapping[int, Mapping[str, ExpertOpinion]],
) -> dict[str, int]:
    """Per-expert count of adjacent direction reversals (§5.4.4).

    ``abstain`` is skipped rather than used to break the sequence: an expert that
    went approve -> reject -> abstain -> approve did reverse twice, and the
    abstention does not undo that.

    A reversal needs two consecutive *steps*, so no expert can be flagged before
    round 3. Only experts with at least one reversal are returned: the recorded
    predicate is the plain reading of §5.4.4 ("决策序列出现相邻的反向变化"),
    while whether that should *trigger* anything is Q-17's open question — hence
    the counts, which are what a threshold would have to be calibrated against.
    """
    sequences: dict[str, list[str]] = {}
    for round_no in sorted(history):
        for expert, opinion in history[round_no].items():
            sequences.setdefault(expert, []).append(opinion.decision)

    reversals: dict[str, int] = {}
    for expert, decisions in sequences.items():
        ranks = [_DECISION_RANK[d] for d in decisions if d in _DECISION_RANK]
        count = sum(
            1
            for first, second, third in zip(ranks, ranks[1:], ranks[2:])
            if (second - first) * (third - second) < 0
        )
        if count:
            reversals[expert] = count
    return reversals


# --------------------------------------------------------------------------- #
# Stage 5 — rendering
# --------------------------------------------------------------------------- #


def render_markdown(
    *,
    request: str,
    grounding: Grounding,
    opinions: dict[str, ExpertOpinion],
    score: float,
    threshold: float,
    status: str,
    rounds: int,
    evidence_gists: dict[str, str],
    oscillation: Mapping[str, int] | None = None,
    evidence_requests: Sequence[EvidenceRequest] = (),
    conditions: Sequence[str] = (),
    review_reason: ReviewReason | None = None,
) -> str:
    lines = [
        "# 工程变更方案",
        "",
        f"- 请求：{request}",
        f"- 依据等级：{'history_backed' if grounding.has_history else 'knowledge_based'}",
        f"- 支持度：{score:.2f}（描述性统计，不单独决定判定：明确通过还需「无反对」）",
        f"- 状态：{status}",
        f"- 轮次：{rounds}",
    ]
    # 「交人原因」必须出现在报告里（§5.6.1「交给人时要说什么」）：状态值只说「交人」，接手的人
    # 需要知道**因为什么**交人——专家的分歧、还是证据缺口、还是有人缺席（D-96）。
    if review_reason is not None:
        lines.append(f"- 交人原因：{REVIEW_REASON_LABELS[review_reason]}")
    if oscillation:
        # Recorded, so it must be *visible* (P5: no silent behaviour) — and
        # labelled as not affecting the verdict, because nothing acts on it yet.
        flips = "；".join(f"{e}（{n} 次反向）" for e, n in sorted(oscillation.items()))
        lines.append(f"- 迭代振荡（仅记录，未影响本次判定）：{flips}")
    lines += ["", "## 各专业意见", ""]
    for expert in sorted(opinions):
        op = opinions[expert]
        name = DISCIPLINE_NAMES.get(expert, expert)
        lines.append(f"### {expert} {name} — {op.decision}")
        lines.append("")
        lines.append(f"- 依据：{op.basis}")
        lines.append(f"- 风险等级：{op.risk_level}")
        if op.rationale:
            lines.append(f"- 理由：{op.rationale}")
        if op.constraints:
            lines.append("- 约束：" + "；".join(op.constraints))
        if op.uncertainties:
            lines.append("- 不确定：" + "；".join(op.uncertainties))
        lines.append(f"- 引用证据：{', '.join(sorted(op.evidence_ids))}")
        lines.append("")

    if conditions:
        # 前置条件是交付物的一部分（D-95）：交付时它是施工前必须满足的条件，交人时它是裁定的
        # 依据。两种终态都要带——否则「有条件通过」在交付物里会变成「无条件通过」。
        lines.append("## 前置条件（施工/采购前必须满足）")
        lines.append("")
        for item in conditions:
            lines.append(f"- {item}")
        lines.append("")

    if evidence_gists:
        lines.append("## 证据清单")
        lines.append("")
        for eid in sorted(evidence_gists):
            lines.append(f"- `{eid}`：{evidence_gists[eid]}")
        lines.append("")

    if evidence_requests:
        # 缺口清单第一次进入**结构化契约**（RunResult.evidence_requests），报告里也要看得见：
        # 「缺什么」是这次评审最可执行的产出，不该只沉在 uncertainties 的散文里（D-94）。
        lines.append("## 需要补充的证据")
        lines.append("")
        for gap in evidence_requests:
            lines.append(f"- [{gap.scope}] {gap.query}")
        lines.append("")
        lines.append("（检索端 `RetrieverPort.search()` 尚未实现：以上请求已登记，落地后即刻生效。）")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


async def run(
    run_input: RunInput,
    ctx: _RunContext,
    *,
    max_rounds: int | None = None,
    threshold: float | None = None,
    min_effective_experts: int | None = None,
    top_k: int | None = None,
    weights: dict[str, float] | None = None,
) -> RunResult:
    cfg = default_settings
    max_rounds = max_rounds or cfg.max_rounds
    threshold = cfg.consensus_threshold if threshold is None else threshold
    min_effective = (
        cfg.min_effective_experts if min_effective_experts is None else min_effective_experts
    )
    top_k = top_k or cfg.top_k

    warnings: list[str] = []
    ctx.events.emit("run_started", request=run_input.request, max_rounds=max_rounds)

    # --- stage 1: intent ------------------------------------------------- #
    ctx.events.emit("node_started", node="intent_complete")
    intent, _entities, _expansion = await complete_intent(run_input, ctx)

    # --- stage 2: baseline ----------------------------------------------- #
    ctx.events.emit("node_started", node="prefetch")
    bundle = await prefetch_baseline(intent, ctx, top_k)
    grounding = assess_grounding(bundle)
    ctx.events.emit(
        "grounding_assessed",
        basis=grounding.basis,
        has_history=grounding.has_history,
        note=grounding.note,
    )

    # --- stage 3: 候选专家骨架（规则）+ 守卫/运行时装配 ------------------- #
    ctx.events.emit("node_started", node="project")
    memory = MemoryService(ctx.registry)
    # 守卫是全局状态的唯一写者；runtime 是唯一执行入口（D-71 / §3.1）。
    guard = Guard(ctx, memory=memory)
    skeleton = RuleSkeleton()
    # 人类在 HITL 里改过的专家集优先于规则表（D-19：用户调整不受限但留痕）。
    active_override: list[str] | None = None

    if not grounding.has_history:
        warnings.append("no_history")
        review = HumanReviewRequest(
            understood_request=intent.normalized_request,
            retrieved_evidence=ctx.registry.refs(bundle.baseline_ids)[:8],
            missing=["历史变更案例", "同类组件处置经验"],
            candidate_experts=select_experts(
                intent.normalized_request, intent.historical_disciplines
            ),
            questions=["是否确认按上述专家范围评估？", "是否有可补充的背景信息？"],
        )
        ctx.events.emit("degradation", level="knowledge_based", reason="no_history")
        if ctx.ask_human is not None:
            human_decision = await ctx.ask_human(review)
            if not human_decision.proceed:
                ctx.events.emit("run_aborted_by_user")
                return RunResult(
                    request=run_input.request,
                    normalized_request=intent.normalized_request,
                    conclusion="用户中止了本次评估。",
                    grounding=grounding,
                    warnings=tuple(warnings),
                )
            # 人类提供的事实登记为**一等证据**（source="human"），可回溯性不破例；
            # 而写 registry 的只有守卫（D-28 / D-71）。
            guard.register_human_facts(f.raw_text for f in human_decision.provided_facts)
            chosen = [
                e
                for e in EXPERT_IDS
                if e in set(human_decision.approved_experts) | set(human_decision.extra_experts)
            ]
            for excluded in human_decision.excluded_by_user:
                warnings.append(f"excluded_by_user:{excluded}")
            if chosen:
                active_override = chosen
            ctx.events.emit(
                "human_decision_applied",
                approved=human_decision.approved_experts,
                extra=human_decision.extra_experts,
                excluded=human_decision.excluded_by_user,
                facts=[f.raw_text for f in human_decision.provided_facts],
            )
        else:
            warnings.append("no_history_headless")

    # ═══ 中段：元智能体的 think-act-observe 循环（内环，§5.4.1） ═══════════ #
    # 裁定权在外环：共识与停滞判定以回调注入（D-73），因此内环不必 import 外环。
    runtime = AgentRuntime(ctx, guard=guard, memory=memory, run_id=ctx.run_id)
    loop = await runtime.run_meta(
        META_SPEC,
        think=partial(
            skeleton.think,
            request=intent.normalized_request,
            historical_disciplines=intent.historical_disciplines,
            active_override=active_override,
            weights_override=weights,
        ),
        closing=skeleton.closing,
        request=intent.normalized_request,
        sub_questions=intent.sub_questions,
        max_rounds=max_rounds,
        consensus_fn=partial(consensus, threshold=threshold, min_effective=min_effective),
        # 「因为什么交人」同样是裁定，因此与共识/停滞一样**由外环注入**（D-73 / D-96）：
        # 内环只认三个状态值，不 import `workflow`，单向依赖在目录层面保持成立。
        reason_fn=partial(manual_review_reason, min_effective=min_effective),
        # 证据缺口由专家写好的待补清单确定性派生（D-94），在 observe 阶段经 act 通道提交。
        gap_fn=partial(skeleton.gaps, request=intent.normalized_request),
        stall_fn=detect_stall,
    )
    warnings.extend(loop.warnings)
    stall = loop.stall

    # 证据请求：管道已通，但检索端口还没有 `search()`（docs/rag.md §9 的双接口只实现了
    # prefetch）。**不静默**：显式标注未满足并落事件（P5）。
    if guard.evidence_requests:
        if hasattr(ctx.retriever, "search"):
            ctx.events.emit("evidence_request_deferred", count=len(guard.evidence_requests))
        else:
            warnings.append("evidence_request_unsatisfied")
            ctx.events.emit(
                "degradation",
                level="evidence_request",
                reason="检索端口未实现 search()，本轮证据请求无法满足（docs/rag.md §9）",
            )

    # --- stage 4：循环已由 AgentsRuntime 执行，见中段 --------------------- #

    # ═══ 后段：确定性收尾（外环） ══════════════════════════════════════════ #
    # 振荡：只记录、不驱动控制流（D-81）。逐轮决策矩阵本来就在事件日志的
    # `round_finished` 里，这里只是把它算成一个可标定的分布。
    oscillation = detect_oscillations(loop.opinions_by_round)
    stall.oscillating_experts = sorted(oscillation)
    stall.reversals = dict(sorted(oscillation.items()))
    if oscillation:
        ctx.events.emit(
            "oscillation_observed",
            experts=stall.oscillating_experts,
            reversals=stall.reversals,
        )

    evidence_gists = {
        ref.evidence_id: ref.gist for ref in ctx.registry.refs(ctx.registry.all_ids())
    }
    # 前置条件是**交付物的一部分**（D-95）：交付时它是施工前必须满足的条件，交人时它是裁定的
    # 依据。两种终态都带——否则「有条件通过」在交付物里会变成「无条件通过」。
    conditions = collect_hard_constraints(loop.latest.values())
    conclusion = render_markdown(
        request=intent.normalized_request,
        grounding=grounding,
        opinions=loop.latest,
        score=loop.score,
        threshold=threshold,
        status=loop.status,
        rounds=loop.rounds,
        evidence_gists=evidence_gists,
        oscillation=stall.reversals,
        evidence_requests=loop.evidence_requests,
        conditions=conditions,
        review_reason=loop.review_reason,
    )

    # 保证等级必须反映**专家实际用的依据**，而不只是「检索有没有命中」。真实 run 里出现过
    # 「4 位专家全凭领域通识判断、系统却报 history_backed」——那等于替一次通识推断背书。
    # 规则与共识分同口径：以**配置权重和**为分母，声明 basis="knowledge" 的权重占比 > 0.5
    # 即降级，并在 inference_basis 里写明原因。
    knowledge_share = _knowledge_share(loop.latest, loop.weights)
    downgraded = grounding.has_history and knowledge_share > 0.5
    # 本轮没有任何专业判断（专家一致判断证据不足）时**不得**声称 history_backed：检索命中了
    # 历史案例，但没有人用它作出结论（D-93 的连带要求，与 basis 降级同一理由）。判据从状态取值
    # 改成**原因标签**（D-96）——证据缺口如今是 `manual_review` 的一种原因，不再是独立状态。
    gap_only = loop.review_reason == "evidence_gap"
    level = (
        "knowledge_based"
        if (not grounding.has_history or downgraded or gap_only)
        else "history_backed"
    )
    inference_basis: list[str] = []
    if not grounding.has_history:
        inference_basis = [grounding.note]
    elif gap_only:
        inference_basis = ["本轮无任何专业判断（专家一致判断证据不足）：不得声称 history_backed"]
    elif downgraded:
        reason = (
            f"{knowledge_share:.0%} 的权重来自声明 basis=knowledge 的专家："
            "本轮证据未支撑其技术细节，故降为 knowledge_based"
        )
        inference_basis = [reason]
    assurance = AssuranceLevel(
        level=level,
        supporting_evidence=sorted(bundle.baseline_ids) if level == "history_backed" else [],
        inference_basis=inference_basis,
    )
    ctx.events.emit(
        "run_finished",
        status=loop.status,
        consensus_score=loop.score,
        assurance=assurance.level,
        usage=loop.usage.model_dump(),
    )

    return RunResult(
        request=run_input.request,
        normalized_request=intent.normalized_request,
        conclusion=conclusion,
        evidence_ids=sorted(ctx.registry.all_ids()),
        active_experts=loop.active,
        consensus_score=loop.score,
        consensus_status=loop.status,  # type: ignore[arg-type]
        review_reason=loop.review_reason,  # type: ignore[arg-type]
        stall=stall,
        evidence_requests=loop.evidence_requests,
        conditions=conditions,
        grounding=grounding,
        assurance=assurance,
        usage=loop.usage,
        warnings=tuple(warnings),
        rounds=loop.rounds,
    )


def new_run_context(**kwargs: object) -> _RunContext:
    """Convenience factory used by the CLI and tests."""
    kwargs.setdefault("run_id", uuid.uuid4().hex[:12])
    return _RunContext(**kwargs)  # type: ignore[arg-type]