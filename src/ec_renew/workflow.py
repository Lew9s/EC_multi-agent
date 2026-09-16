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
from .agents.memory import MemoryService
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
    ExpertOpinion,
    GraphExpansion,
    Grounding,
    HumanReviewRequest,
    IntentCompletion,
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
) -> tuple[float, int, str]:
    """Denominator is the configured weight sum (D-38).

    An abstaining expert contributes 0 to the numerator but keeps its weight in
    the denominator, so abstaining can only *lower* the score.
    """
    effective = sum(1 for op in opinions.values() if op.decision != "abstain")
    if effective < min_effective:
        return 0.0, effective, "manual_review"

    numerator = sum(
        weights.get(expert, 0.0) * DECISION_SCORES[op.decision]
        for expert, op in opinions.items()
    )
    denominator = sum(weights.get(expert, 0.0) for expert in weights)
    score = round(numerator / denominator, 4) if denominator else 0.0
    return score, effective, "approved" if score >= threshold else "retry"


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
) -> str:
    lines = [
        "# 工程变更方案",
        "",
        f"- 请求：{request}",
        f"- 依据等级：{'history_backed' if grounding.has_history else 'knowledge_based'}",
        f"- 共识分：{score:.2f}（阈值 {threshold:.2f}）",
        f"- 状态：{status}",
        f"- 轮次：{rounds}",
    ]
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

    if evidence_gists:
        lines.append("## 证据清单")
        lines.append("")
        for eid in sorted(evidence_gists):
            lines.append(f"- `{eid}`：{evidence_gists[eid]}")
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
    )

    # 保证等级必须反映**专家实际用的依据**，而不只是「检索有没有命中」。真实 run 里出现过
    # 「4 位专家全凭领域通识判断、系统却报 history_backed」——那等于替一次通识推断背书。
    # 规则与共识分同口径：以**配置权重和**为分母，声明 basis="knowledge" 的权重占比 > 0.5
    # 即降级，并在 inference_basis 里写明原因。
    knowledge_share = _knowledge_share(loop.latest, loop.weights)
    downgraded = grounding.has_history and knowledge_share > 0.5
    level = "knowledge_based" if (not grounding.has_history or downgraded) else "history_backed"
    inference_basis: list[str] = []
    if not grounding.has_history:
        inference_basis = [grounding.note]
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
        stall=stall,
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