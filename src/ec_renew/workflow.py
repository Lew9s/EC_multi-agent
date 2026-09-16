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

import asyncio
import uuid
from collections.abc import Mapping, Sequence

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
    ExpertTask,
    GraphExpansion,
    Grounding,
    HumanReviewRequest,
    IntentCompletion,
    RunInput,
    RunResult,
    StallReport,
    SubQuestion,
    Usage,
)
from .errors import BudgetExceeded, InvariantViolation, StepLimitExceeded
from .experts import abstain_opinion, run_expert, select_experts
from .memory import MemoryService
from .ports import RunContext as _RunContext

MAX_QUERIES = 5
UNIFORM_WEIGHT = 1.0


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


async def _run_one(task: ExpertTask, ctx: _RunContext, allowed: Sequence[str]) -> tuple[ExpertOpinion, Usage]:
    """Expected failures become values; fatal errors stay exceptions.

    This conversion point is what decides cancellation semantics: because a
    timeout/transport error is turned into an abstain *value*, ``TaskGroup``
    never sees it and the other experts are allowed to finish.
    """
    try:
        async with asyncio.timeout(task.budget.timeout_s):
            return await run_expert(task, ctx.llm, allowed, events=ctx.events)
    except asyncio.CancelledError:
        raise
    except (InvariantViolation, BudgetExceeded, StepLimitExceeded):
        raise
    except Exception as exc:  # noqa: BLE001 — 有意的「异常转状态」点，见下方说明
        # Expected failures become values; fatal errors stay exceptions.
        # 这是全流程唯一的宽泛捕获点：超时/传输错误在这里变成 abstain 值，
        # TaskGroup 因此看不到它，兄弟专家得以跑完（docs/design.md §7.6）。
        ctx.events.emit(
            "failure",
            node="expert",
            expert=task.expert,
            cause_type=type(exc).__name__,
            message=str(exc)[:200],
        )
        return abstain_opinion(task.expert, allowed, f"{type(exc).__name__}"), Usage(calls=1)


async def dispatch(
    tasks: Sequence[ExpertTask], ctx: _RunContext
) -> tuple[dict[str, ExpertOpinion], Usage]:
    allowed = ctx.registry.all_ids()
    async with asyncio.TaskGroup() as group:
        running = [group.create_task(_run_one(task, ctx, allowed)) for task in tasks]

    opinions: dict[str, ExpertOpinion] = {}
    usage = Usage()
    # Sorted merge -> order independent, identical on replay.
    for task, runner in sorted(zip(tasks, running), key=lambda pair: pair[0].expert):
        opinion, spent = runner.result()
        opinions[task.expert] = opinion
        usage = usage + spent
    return opinions, usage


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
    total_usage = Usage()
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

    # --- stage 3: select experts (rule table) ---------------------------- #
    active = select_experts(intent.normalized_request, intent.historical_disciplines)

    ctx.events.emit("node_started", node="project")
    human_decision = None

    if not grounding.has_history:
        warnings.append("no_history")
        review = HumanReviewRequest(
            understood_request=intent.normalized_request,
            retrieved_evidence=ctx.registry.refs(bundle.baseline_ids)[:8],
            missing=["历史变更案例", "同类组件处置经验"],
            candidate_experts=active,
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
            # User-supplied facts become first-class evidence (source="human"),
            # so the traceability rule holds without exception.
            for fact in human_decision.provided_facts:
                ctx.registry.register(source="human", content=fact.raw_text)
            chosen = [
                e
                for e in EXPERT_IDS
                if e in set(human_decision.approved_experts) | set(human_decision.extra_experts)
            ]
            for excluded in human_decision.excluded_by_user:
                warnings.append(f"excluded_by_user:{excluded}")
            if chosen:
                active = chosen
            ctx.events.emit(
                "human_decision_applied",
                approved=human_decision.approved_experts,
                extra=human_decision.extra_experts,
                excluded=human_decision.excluded_by_user,
                facts=[f.raw_text for f in human_decision.provided_facts],
            )
        else:
            warnings.append("no_history_headless")

    weights = weights or {expert: UNIFORM_WEIGHT for expert in active}
    memory = MemoryService(ctx.registry)
    ctx.events.emit("experts_selected", active=active, weights=weights)

    # --- stage 4: consensus rounds --------------------------------------- #
    opinions_by_round: dict[int, dict[str, ExpertOpinion]] = {}
    latest: dict[str, ExpertOpinion] = {}
    previous: dict[str, ExpertOpinion] = {}
    previous_baseline: list[str] = []
    stall = StallReport()
    score, effective, status = 0.0, 0, "retry"
    round_no = 0

    for round_no in range(1, max_rounds + 1):
        # Recomputed every round, not once outside the loop: evidence registered
        # *during* a run (human facts today, EvidenceRequest later) belongs in
        # the next frozen baseline. Freezing one snapshot up front also made the
        # "no new evidence" half of the stall test vacuously true.
        baseline = ctx.registry.all_ids()
        ctx.registry.freeze(round_no, baseline)
        ctx.events.emit("round_started", round=round_no, experts=active)

        tasks: list[ExpertTask] = []
        for expert in active:
            task, record = memory.project(
                expert=expert,
                request=intent.normalized_request,
                round_no=round_no,
                sub_questions=intent.sub_questions,
                previous=latest.get(expert),
                opinions=latest.values(),
                consensus_score=score,
                dissent_count=sum(1 for op in latest.values() if op.decision != "approve"),
            )
            tasks.append(task)
            ctx.projections.append(record)
            ctx.events.emit(
                "projected",
                round=round_no,
                expert=expert,
                policy=record.policy.value,
                disclosed=len(record.disclosed_claim_ids),
            )

        round_opinions, spent = await dispatch(tasks, ctx)
        total_usage = total_usage + spent
        opinions_by_round[round_no] = round_opinions
        latest = dict(round_opinions)

        score, effective, status = consensus(round_opinions, weights, threshold, min_effective)
        ctx.events.emit(
            "round_finished",
            round=round_no,
            consensus_score=score,
            effective_experts=effective,
            status=status,
            decisions={e: op.decision for e, op in sorted(round_opinions.items())},
        )
        if status != "retry":
            break

        # --- fixed point (D-81) ------------------------------------------- #
        # Only meaningful from round 2 on: round 1 -> 2 also changes `mode`,
        # `revision` and `cross_agent` structurally, so their prompts differ for
        # reasons that have nothing to do with information gain.
        if previous and detect_stall(previous, round_opinions, previous_baseline, baseline):
            stall = StallReport(
                stalled=True,
                detected_at_round=round_no,
                skipped_rounds=max_rounds - round_no,
                unchanged_experts=sorted(round_opinions),
            )
            ctx.events.emit("stalled", round=round_no, experts=stall.unchanged_experts)
            ctx.events.emit("stall_skipped", skipped_rounds=stall.skipped_rounds)
            # Not `manual_review`: no expert is still disagreeing, the process
            # has simply stopped producing information (§5.6.1).
            status = "stalled"
            warnings.append("stalled")
            break

        previous = dict(round_opinions)
        previous_baseline = baseline

    if status == "retry":
        status = "manual_review"
        warnings.append("max_rounds_reached")

    # --- oscillation: recorded, never acted upon (D-81) ------------------- #
    # Computed once at the end over the whole decision history. The per-round
    # decision matrix is already in the event log (`round_finished`), so this
    # records the interpretation rather than the raw data — and it deliberately
    # steers nothing: no expert is skipped and no round is cut short.
    oscillation = detect_oscillations(opinions_by_round)
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
        opinions=latest,
        score=score,
        threshold=threshold,
        status=status,
        rounds=round_no,
        evidence_gists=evidence_gists,
        oscillation=stall.reversals,
    )

    assurance = AssuranceLevel(
        level="history_backed" if grounding.has_history else "knowledge_based",
        supporting_evidence=sorted(bundle.baseline_ids) if grounding.has_history else [],
        inference_basis=[] if grounding.has_history else [grounding.note],
    )
    ctx.events.emit(
        "run_finished",
        status=status,
        consensus_score=score,
        assurance=assurance.level,
        usage=total_usage.model_dump(),
    )

    return RunResult(
        request=run_input.request,
        normalized_request=intent.normalized_request,
        conclusion=conclusion,
        evidence_ids=sorted(ctx.registry.all_ids()),
        active_experts=active,
        consensus_score=score,
        consensus_status=status,  # type: ignore[arg-type]
        stall=stall,
        grounding=grounding,
        assurance=assurance,
        usage=total_usage,
        warnings=tuple(warnings),
        rounds=round_no,
    )


def new_run_context(**kwargs: object) -> _RunContext:
    """Convenience factory used by the CLI and tests."""
    kwargs.setdefault("run_id", uuid.uuid4().hex[:12])
    return _RunContext(**kwargs)  # type: ignore[arg-type]