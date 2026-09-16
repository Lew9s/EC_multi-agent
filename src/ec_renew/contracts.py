"""Cross-layer contracts.

This module is the ONLY interface between the outer loop (workflow) and the
inner loop (agents). It imports nothing from the business modules, so it can
never be dragged into a dependency cycle.

Conventions
-----------
* Anything that crosses a module boundary is a model here, never a bare dict.
* Collections written by parallel branches are **keyed**, never ordered lists
  (order is a rendering concern, produced by sorting at render time).
* Every fact-bearing object carries ``evidence_ids``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

EXPERT_IDS: tuple[str, ...] = ("E01", "E02", "E03", "E04", "E05", "E06")

DISCIPLINE_NAMES: dict[str, str] = {
    "E01": "结构设计",
    "E02": "舾装工艺",
    "E03": "质量规范",
    "E04": "电气系统",
    "E05": "轮机系统",
    "E06": "材料与焊接",
}

DECISION_SCORES: dict[str, float] = {
    "approve": 1.0,
    "revise": 0.5,
    "reject": 0.0,
    "abstain": 0.0,
}

EvidenceSource = Literal["graph", "text", "human", "standard"]
ReviewDecision = Literal["approve", "revise", "reject", "abstain"]
Autonomy = Literal["L0", "L1", "L2"]


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


class Evidence(BaseModel):
    """A single retrievable fact, stored once in the run-scoped registry."""

    evidence_id: str
    source: EvidenceSource
    content: str
    entity_kinds: list[str] = Field(default_factory=list)
    entity_keys: list[str] = Field(default_factory=list)
    disciplines: list[str] = Field(default_factory=list)
    group_keys: list[str] = Field(default_factory=list)
    source_file: str = ""
    score: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvidenceRef(BaseModel):
    """Lightweight handle handed to agents (keeps their context small)."""

    evidence_id: str
    gist: str
    source: EvidenceSource = "graph"


class EvidenceMeta(BaseModel):
    """Structured metadata handed to the meta agent.

    ``disciplines`` is the key field: it lets the meta agent decide which
    experts to activate from data instead of guessing from prose.
    """

    evidence_id: str
    entity_kinds: list[str] = Field(default_factory=list)
    entity_keys: list[str] = Field(default_factory=list)
    disciplines: list[str] = Field(default_factory=list)
    group_keys: list[str] = Field(default_factory=list)
    source: EvidenceSource = "graph"
    score: float = 0.0
    # Full text. The registry is content-addressed, so it recomputes the id
    # from this string; ``evidence_id`` above is only a retriever-side label.
    content: str = ""


class EvidenceBundle(BaseModel):
    """Result of ``Retriever.prefetch`` — the frozen baseline of one round."""

    round: int = 0
    baseline_ids: list[str] = Field(default_factory=list)
    items: list[EvidenceMeta] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _baseline_matches_items(self) -> EvidenceBundle:
        known = {item.evidence_id for item in self.items}
        self.baseline_ids = [eid for eid in self.baseline_ids if eid in known]
        return self


class EvidenceView(BaseModel):
    """What one expert sees this round."""

    round: int = 0
    baseline_ids: list[str] = Field(default_factory=list)
    delta_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Intent
# --------------------------------------------------------------------------- #


class EntityRef(BaseModel):
    name: str
    kind: Literal["COMPONENT", "DEPARTMENT", "REASON", "TIME_POINT", "UNKNOWN"] = "UNKNOWN"
    in_graph: bool = False
    graph_key: str | None = None


class GraphExpansion(BaseModel):
    """graph-grounded completion of an abstract user request."""

    parent_components: list[str] = Field(default_factory=list)
    historical_departments: list[str] = Field(default_factory=list)
    historical_disciplines: list[str] = Field(default_factory=list)
    similar_case_ids: list[str] = Field(default_factory=list)


class SubQuestion(BaseModel):
    text: str
    discipline: str = "E01"
    priority: float = 0.5


class IntentCompletion(BaseModel):
    raw_request: str
    normalized_request: str
    mentioned_entities: list[EntityRef] = Field(default_factory=list)
    graph_expansion: GraphExpansion = Field(default_factory=GraphExpansion)
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    query_set: list[str] = Field(default_factory=list)
    provenance: Literal["rule", "graph", "llm"] = "rule"

    @property
    def historical_disciplines(self) -> list[str]:
        return list(dict.fromkeys(self.graph_expansion.historical_disciplines))


class Grounding(BaseModel):
    """Whether this run rests on real historical cases."""

    basis: Literal["history", "knowledge"] = "knowledge"
    case_ids: list[str] = Field(default_factory=list)
    note: str = ""

    @property
    def has_history(self) -> bool:
        return self.basis == "history" and bool(self.case_ids)


class AssuranceLevel(BaseModel):
    """How well-supported a conclusion is. Two levels only."""

    level: Literal["history_backed", "knowledge_based"] = "knowledge_based"
    supporting_evidence: list[str] = Field(default_factory=list)
    inference_basis: list[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Cross-agent projection
# --------------------------------------------------------------------------- #


class Claim(BaseModel):
    """Smallest unit that may cross an agent boundary.

    ``claim`` is copied verbatim, never re-summarised, and length-capped, so a
    sub agent cannot smuggle instructions into another sub agent's prompt.
    """

    claim: str = Field(max_length=200)
    condition: str | None = None
    evidence_ids: list[str] = Field(min_length=1)
    discipline: str = "E01"


class DisclosurePolicy(str, Enum):
    NONE = "none"
    ANONYMOUS_CLAIMS = "anon_claims"

    @classmethod
    def for_round(cls, round: int) -> DisclosurePolicy:
        """Derived from the round, never chosen by an LLM."""
        return cls.NONE if round <= 1 else cls.ANONYMOUS_CLAIMS


class CrossAgentInfo(BaseModel):
    """What the meta agent is allowed to show a sub agent about its peers."""

    anonymous_claims: list[Claim] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    consensus_score: float = 0.0
    dissent_count: int = 0


class ReviewFeedback(BaseModel):
    """Delphi-style controlled feedback: aggregates only, never arguments."""

    round: int = 1
    consensus_score: float = 0.0
    dissent_count: int = 0
    anonymous_dissent: list[str] = Field(default_factory=list)


class RevisionContext(BaseModel):
    own_previous: ExpertOpinion
    new_evidence_ids: list[str] = Field(default_factory=list)
    feedback: ReviewFeedback | None = None


class ProjectionRecord(BaseModel):
    """Audit trail: what was disclosed to whom this round."""

    round: int
    expert: str
    policy: DisclosurePolicy
    disclosed_claim_ids: list[str] = Field(default_factory=list)
    hard_constraint_ids: list[str] = Field(default_factory=list)


class AgentBudget(BaseModel):
    max_steps: int = 1
    max_tokens: int = 0
    timeout_s: float = 120.0


# --------------------------------------------------------------------------- #
# Expert hand-off (outer loop -> inner loop and back)
# --------------------------------------------------------------------------- #


class ExpertTask(BaseModel):
    mode: Literal["initial", "revise"] = "initial"
    expert: str
    request: str
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    evidence_view: EvidenceView = Field(default_factory=EvidenceView)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    autonomy: Autonomy = "L0"
    budget: AgentBudget = Field(default_factory=AgentBudget)
    round: int = 1
    revision: RevisionContext | None = None
    cross_agent: CrossAgentInfo | None = None

    @model_validator(mode="after")
    def _check_revision_ownership(self) -> ExpertTask:
        if self.revision is not None:
            if self.revision.own_previous.expert != self.expert:
                raise ValueError("revision.own_previous 必须属于该专家本人")
        else:
            if self.mode == "revise":
                raise ValueError("mode='revise' 时必须提供 revision")
        return self


class ExpertOpinion(BaseModel):
    expert: str
    decision: ReviewDecision
    rationale: str = ""
    evidence_ids: list[str] = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"
    confidence: float = Field(default=0.0, ge=0, le=1)  # 仅记录，不参与计分
    partial: bool = False
    assurance: AssuranceLevel = Field(default_factory=AssuranceLevel)

    @computed_field
    @property
    def score(self) -> float:
        return DECISION_SCORES[self.decision]


class Disagreement(BaseModel):
    experts: tuple[str, str]
    kind: Literal["judgment", "evidence", "mixed"] = "judgment"
    evidence_overlap: float = 0.0
    divergent_evidence: list[str] = Field(default_factory=list)


class ActivationPlan(BaseModel):
    active_experts: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    evidence_scope: dict[str, list[str]] = Field(default_factory=dict)
    rationale: str = ""
    cross_domain_flags: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Session (multi-turn)
# --------------------------------------------------------------------------- #


class TurnSummary(BaseModel):
    turn: int
    user_request: str
    conclusion: str
    confirmed_experts: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    assurance: AssuranceLevel = Field(default_factory=AssuranceLevel)
    open_threads: list[str] = Field(default_factory=list)


class SessionSnapshot(BaseModel):
    """Immutable copy handed to a run. A run can never mutate the session."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    anchor_request: str
    recent_turns: tuple[TurnSummary, ...] = ()
    user_constraints: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Run I/O
# --------------------------------------------------------------------------- #


class RunInput(BaseModel):
    request: str
    session: SessionSnapshot = Field(
        default_factory=lambda: SessionSnapshot(session_id="default", anchor_request="")
    )


class Usage(BaseModel):
    """Additive — used as a reducer, so it must be commutative+associative."""

    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            calls=self.calls + other.calls,
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
        )


class LLMResult(BaseModel):
    content: str
    reasoning: str = ""
    model: str = ""
    cached: bool = False
    usage: Usage = Field(default_factory=Usage)


class RunResult(BaseModel):
    request: str
    normalized_request: str
    conclusion: str
    evidence_ids: list[str] = Field(default_factory=list)
    active_experts: list[str] = Field(default_factory=list)
    consensus_score: float = 0.0
    consensus_status: Literal["approved", "manual_review"] = "manual_review"
    grounding: Grounding = Field(default_factory=Grounding)
    assurance: AssuranceLevel = Field(default_factory=AssuranceLevel)
    usage: Usage = Field(default_factory=Usage)
    warnings: tuple[str, ...] = ()
    rounds: int = 0


class FailureEvent(BaseModel):
    code: str
    node: str
    round: int = 0
    cause_type: str = ""
    message: str = ""
    retryable: bool = False
    attempt: int = 1

# --------------------------------------------------------------------------- #
# Front-loaded human-in-the-loop (no-history branch)
# --------------------------------------------------------------------------- #


class HumanReviewRequest(BaseModel):
    """Shown to the user when the run has no historical grounding.

    Must present the system's own understanding, what was retrieved, and what
    is missing — otherwise "let the user decide" is an uninformed decision.
    """

    reason: Literal["no_history"] = "no_history"
    understood_request: str
    retrieved_evidence: list[EvidenceRef] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    candidate_experts: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)


class HumanProvidedFact(BaseModel):
    """User-supplied fact. ``raw_text`` is never rewritten (audit rule §5.5.3)."""

    raw_text: str
    parsed: dict | None = None


class HumanDecision(BaseModel):
    proceed: bool = True
    approved_experts: list[str] = Field(default_factory=list)
    extra_experts: list[str] = Field(default_factory=list)
    excluded_by_user: list[str] = Field(default_factory=list)
    provided_facts: list[HumanProvidedFact] = Field(default_factory=list)
    note: str = ""