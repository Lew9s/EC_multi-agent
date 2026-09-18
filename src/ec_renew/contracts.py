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

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

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
# Cross-agent text bounds (D-77)
# --------------------------------------------------------------------------- #
# Why these exist: ``Claim.claim`` / ``Claim.condition`` / ``ExpertOpinion.
# constraints`` are pasted **verbatim** into *other* agents' prompts (§8.5.3),
# and ``constraints`` reaches everyone unconditionally (D-56). Without bounds a
# single expert can either inject a fake section heading or blow up every
# peer's context.
#
# These are module constants, not ``Settings`` fields, on purpose: AGENTS.md
# §3.5 requires a calibration basis for anything configurable, and we have
# none for these numbers yet. They are provisional — see docs/design.md §12
# item 1e.
#
# Provisional values, not calibrated:
MAX_CLAIM_CHARS = 200
MAX_CONDITION_CHARS = 200
MAX_CONSTRAINT_CHARS = 200
MAX_CONSTRAINTS = 8
MAX_UNCERTAINTY_CHARS = 200
MAX_UNCERTAINTIES = 8
MAX_RATIONALE_CHARS = 500

#: A claim is a *conclusion*: it must say something and it must stay on one
#: line, because it is copied verbatim into a peer's prompt where a newline
#: could fabricate a markdown heading ("## 新指令 …").
ClaimText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_CLAIM_CHARS,
        pattern=r"^[^\r\n]*$",
    ),
]

#: Bounded single-line text for list items and optional fields. Blank values
#: are dropped by the owning model rather than rejected here, so a sloppy model
#: does not lose its whole opinion over an empty string.
SingleLineText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        max_length=MAX_CONSTRAINT_CHARS,
        pattern=r"^[^\r\n]*$",
    ),
]


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


def make_claim_id(claim: str, condition: str | None, evidence_ids: Sequence[str]) -> str:
    """Content-addressed claim identity (D-78).

    The same claim + condition + evidence gets the same id anywhere in a run.
    That is what makes the disclosure graph and the "披露-改变率" metric
    computable (§8.5.6): "which claim changed whose opinion?" becomes a set
    operation instead of a text comparison that any whitespace difference or
    two experts saying the same sentence would break.

    Deliberately **excludes** ``discipline``: including it would give one
    argument two different ids depending on who raised it, and would hide the
    informative case of two disciplines independently raising the same
    constraint.

    This is derived by code and exposed as a computed field, so unlike
    ``expert`` it does not even need overwriting at parse time — a model
    cannot forge what is not an input field.
    """
    payload = json.dumps(
        {"claim": claim, "condition": condition or "", "evidence_ids": sorted(evidence_ids)},
        sort_keys=True,
        ensure_ascii=False,
    )
    return "C-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


class Claim(BaseModel):
    """Smallest unit that may cross an agent boundary.

    ``claim`` is copied verbatim, never re-summarised, and length-capped, so a
    sub agent cannot smuggle instructions into another sub agent's prompt.
    ``claim`` and ``condition`` are also single-line for the same reason: a
    newline would let them fabricate a markdown heading in the peer's prompt
    (D-77).
    """

    claim: ClaimText
    condition: SingleLineText | None = None
    evidence_ids: list[str] = Field(min_length=1)
    discipline: str = "E01"

    @computed_field
    @property
    def claim_id(self) -> str:
        return make_claim_id(self.claim, self.condition, self.evidence_ids)

    @field_validator("condition", mode="before")
    @classmethod
    def _blank_condition_means_absent(cls, value: object) -> object:
        """``condition`` is optional (Q-19): an empty string means "no
        condition", not "malformed output" — do not burn a contract retry."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


class AnonymizedClaim(BaseModel):
    """A claim **as it crosses an agent boundary** (D-77).

    Deliberately has no ``discipline`` field. ``E01``..``E06`` map one-to-one
    onto the six experts, so carrying it would be a de-facto identity
    disclosure — forbidden by D-50 and §8.4.5. Before this type existed, the
    only thing keeping identity out of a peer's prompt was the fact that
    ``render_task`` happened not to print the field.

    The field set *is* the guarantee: adding a field to ``Claim`` (which stays
    inside the run state, where discipline is legitimate research data) does
    not widen what crosses. A test pins the exact field set.
    """

    claim: ClaimText
    condition: SingleLineText | None = None
    evidence_ids: list[str] = Field(min_length=1)

    @computed_field
    @property
    def claim_id(self) -> str:
        """Same id as the originating ``Claim``, since the three source fields
        are identical — that identity is what the disclosure record stores."""
        return make_claim_id(self.claim, self.condition, self.evidence_ids)

    @classmethod
    def from_claim(cls, claim: Claim) -> AnonymizedClaim:
        return cls(
            claim=claim.claim,
            condition=claim.condition,
            evidence_ids=list(claim.evidence_ids),
        )


class DisclosurePolicy(str, Enum):
    NONE = "none"
    ANONYMOUS_CLAIMS = "anon_claims"

    @classmethod
    def for_round(cls, round: int) -> DisclosurePolicy:
        """Derived from the round, never chosen by an LLM."""
        return cls.NONE if round <= 1 else cls.ANONYMOUS_CLAIMS


class CrossAgentInfo(BaseModel):
    """What the meta agent is allowed to show a sub agent about its peers."""

    anonymous_claims: list[AnonymizedClaim] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    consensus_score: float = 0.0
    dissent_count: int = 0


class ReviewFeedback(BaseModel):
    """Delphi-style controlled feedback: aggregates only, never arguments."""

    round: int = 1
    consensus_score: float = 0.0
    dissent_count: int = 0
    # Claim **ids**, not text: this field feeds the disclosure graph (§8.5.6),
    # where a stable identity is required to answer "which claim changed whose
    # opinion?" (D-78). Distinct claims only — two experts raising the same
    # argument share an id on purpose.
    anonymous_dissent_ids: list[str] = Field(default_factory=list)


class RevisionContext(BaseModel):
    own_previous: ExpertOpinion
    new_evidence_ids: list[str] = Field(default_factory=list)
    feedback: ReviewFeedback | None = None


class ProjectionRecord(BaseModel):
    """Audit trail: what was disclosed to whom this round."""

    round: int
    expert: str
    policy: DisclosurePolicy
    #: Claim **ids** (``C-…``), not claim text: §8.5.6 promises the disclosure
    #: graph is reconstructible and usable as a research metric, and both need
    #: a stable identity (D-78).
    disclosed_claim_ids: list[str] = Field(default_factory=list)
    #: Constraints stay as **text**, deliberately asymmetric with claims:
    #: constraints are broadcast unconditionally (D-56) rather than aggregated,
    #: they are already bounded (D-77), and the metric's subject is claims.
    #: Inventing a second id scheme here would be abstraction without a user.
    hard_constraints: list[str] = Field(default_factory=list)


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
    #: 判断依据（§5.7）：证据 / 领域通识 / 两者并重。
    #:
    #: 它由专家**如实声明**，用来给保证等级降级：本轮证据给不出技术细节时，专家仍应基于领域
    #: 通识给出判断（见技能 ``agents/skills/expert_review/SKILL.md``），但那样的判断不该冒充
    #: ``history_backed``。缺省 ``evidence`` 是为了与既有假适配器/夹具保持兼容。
    basis: Literal["evidence", "knowledge", "mixed"] = "evidence"
    #: 弃权的**来源**，由服务端在构造点赋值，模型无法设置（D-93）：
    #:
    #: * ``judgment`` —— 模型在契约内弃权：「本请求我无法给出结论」。这是一次**交付**。
    #: * ``execution_failure`` —— 服务端兜底（超时 / 传输错误 / 契约重试耗尽）。这才是**缺席**。
    #:
    #: 区分它们的理由：两者交给人时的补救动作相反（补证据 vs 重试/换模型），而 quorum 只该拦
    #: 后者——D-37 的原意是「避免缺席被当作通过」，不是「弃权即缺席」。
    abstain_kind: Literal["judgment", "execution_failure"] | None = None
    rationale: str = Field(default="", max_length=MAX_RATIONALE_CHARS)
    evidence_ids: list[str] = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)
    # Bounded count *and* per-item length: `constraints` reaches every peer's
    # prompt unconditionally (D-56), so an unbounded list is both an injection
    # surface and a context bomb.
    constraints: list[SingleLineText] = Field(default_factory=list, max_length=MAX_CONSTRAINTS)
    uncertainties: list[SingleLineText] = Field(
        default_factory=list, max_length=MAX_UNCERTAINTIES
    )
    risk_level: Literal["low", "medium", "high"] = "low"
    confidence: float = Field(default=0.0, ge=0, le=1)  # 仅记录，不参与计分
    #: This opinion is incomplete: either the agent hit its budget, or a field
    #: had to be repaired (truncated) after contract retries were exhausted.
    partial: bool = False
    assurance: AssuranceLevel = Field(default_factory=AssuranceLevel)

    @field_validator("constraints", "uncertainties")
    @classmethod
    def _drop_blank_items(cls, values: list[str]) -> list[str]:
        """A blank item carries no information; dropping it must not cost the
        expert its whole opinion (unlike an over-long one, which is a contract
        violation — see ``experts.repair_opinion``)."""
        return [v for v in values if v]

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
    #: 上一版方案（D-99 / §5.9）：迭代回路要「改文本」，就必须拿得到被改的那一版。
    #: 此前会话只带 `TurnSummary.conclusion` 的散文，不够。
    latest_plan: PlanDraft | None = None


# --------------------------------------------------------------------------- #
# Run I/O
# --------------------------------------------------------------------------- #


class RunInput(BaseModel):
    request: str
    session: SessionSnapshot = Field(
        default_factory=lambda: SessionSnapshot(session_id="default", anchor_request="")
    )
    #: 用户对**上一版方案**的逐条意见（D-99 / §5.9 的迭代回路）。非空即表示这一轮走的是
    #: 「只迭代方案文本」的路径——不重跑专家评审（重评要动事实基线，那是另一条回路）。
    plan_feedback: list[str] = Field(default_factory=list)


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


class LLMCallMeta(BaseModel):
    """Machine-readable metadata for one LLM call (design §12 1f).

    It deliberately does **not** travel inside the message content. It is handed
    to the port out of band so the cache key, the event log and the fake adapter
    can see *which* call this is, without the model having to read a header line
    and without that line perturbing the prompt.

    Why it is more than hygiene: with the round counter out of the prompt, two
    consecutive rounds over an unchanged frozen baseline render a **byte-identical**
    prompt, which is what upgrades §5.4.4's fixed-point argument from "no further
    information gain" to a proof that the next round could only repeat this one.
    It also keeps the prompt prefix stable across rounds, so the provider's
    KV cache can actually hit.
    """

    expert: str = ""
    round: int = 0
    mode: str = ""


class LLMResult(BaseModel):
    content: str
    reasoning: str = ""
    model: str = ""
    cached: bool = False
    usage: Usage = Field(default_factory=Usage)


class StallReport(BaseModel):
    """Iteration-stability findings (§5.4.4, D-81/D-82).

    Two determinations share this report because both come out of the same
    round-by-round data and both answer "is iterating further worth anything":

    * **fixed point** — a round adds no evidence and moves no prompt-affecting
      field, so the *next* round could only repeat it. Reported separately from a
      bare status because the human needs to see that this is "the process has
      stopped producing information", not "the experts still disagree" (§5.6.1).
    * **oscillation** — an expert reversed direction between adjacent rounds.
      Recorded only: the threshold that should *trigger* an action is not
      calibrated yet (Q-17), and a threshold cannot be calibrated without a
      distribution to look at (D-81). Nothing here steers control flow.

    ``reversals`` is that distribution — expert -> number of direction changes —
    so a later change can pick between "≥1 reversal" and "≥2 consecutive
    reversals" from real runs instead of from intuition. The per-round decision
    matrix it is derived from is already in the event log (``round_finished``),
    so this is a convenience view, not the only copy.
    """

    stalled: bool = False
    detected_at_round: int = 0
    skipped_rounds: int = 0
    unchanged_experts: list[str] = Field(default_factory=list)
    # Recorded, never acted upon yet (D-81). `reused_from_round` — the field the
    # interception path will need — is deliberately absent until then: adding it
    # now would imply opinions are being reused, and none are.
    oscillating_experts: list[str] = Field(default_factory=list)
    reversals: dict[str, int] = Field(default_factory=dict)


#: ``manual_review`` 的**原因**（D-96）：状态值只回答「交不交人」，原因回答「交给人时该说什么」。
#: 两者拆开之后，状态机不必为每种情形各长一个取值——原因由专家意见的**构成**确定性判定
#: （`workflow.manual_review_reason`），人可以读报告里的「交人原因」行，机器读 `review_reason`。
#: ``unsupported_plan`` 是**唯一**不来自专家意见构成的一条：方案节点拒绝整份方案时（D-98）由外环
#: 覆写，因此 `manual_review_reason()` 永远不会返回它。
ReviewReason = Literal[
    "quorum", "disagreement", "conditions_only", "evidence_gap", "unsupported_plan"
]

#: 方案骨架的固定章节（**D-98**）。必须是**枚举**：「骨架由契约固定」只有枚举才有可校验的形态——
#: 若让模型自创标题，就无法判断它有没有漏掉「风险与前置条件」这一节。
PlanHeading = Literal["scope", "basis", "execution", "risk", "open_questions", "evidence_index"]

#: 章节 → 渲染用中文标题。放在契约里而不是渲染函数里，是为了让「骨架」只有一个定义处。
PLAN_HEADING_LABELS: dict[str, str] = {
    "scope": "变更范围",
    "basis": "技术依据",
    "execution": "施工与采购要求",
    "risk": "风险与前置条件",
    "open_questions": "待确认事项",
    "evidence_index": "证据索引",
}

#: 缺失即视为方案不合格的章节（Q-30 未定前取最小集：前四节）。
REQUIRED_PLAN_HEADINGS: frozenset[str] = frozenset({"scope", "basis", "execution", "risk"})

#: ``RunResult.plan_source``：方案由模型撰写还是模板拼出。**降级必须显式**（P5）。
PlanSource = Literal["agent", "template"]


class PlanClaim(BaseModel):
    """方案里**需要可回溯**的一条（D-98）。``evidence_ids`` 为空即「无支撑」，契约层直接挡掉。"""

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class PlanSection(BaseModel):
    """一节方案：``body`` 是自由叙述，``claims`` 是需要逐条可回溯的部分。"""

    heading: PlanHeading
    body: str = ""
    claims: list[PlanClaim] = Field(default_factory=list)


class PlanDraft(BaseModel):
    """方案产物（D-98）：**骨架结构化 + 正文自由**。

    为什么这样切：§5.7 的硬约束是「最终方案中任何条目不得无支撑」，而「哪句话靠哪条证据」无法从
    一段自由 Markdown 里机械校验。把需要可回溯的部分（``claims``）与叙述部分（``body``）分开，
    校验就落在骨架上，行文也不被模板腔绑住。
    """

    sections: list[PlanSection] = Field(default_factory=list)
    #: 撰写时做过的假设。它是不带证据的叙述的**唯一出口**——不能进 `claims` 的话必须写在这里被看见。
    assumption_notes: list[str] = Field(default_factory=list)
    #: §5.9 的迭代回路每改一版递增。
    version: int = 1


class RunResult(BaseModel):
    request: str
    normalized_request: str
    conclusion: str
    evidence_ids: list[str] = Field(default_factory=list)
    active_experts: list[str] = Field(default_factory=list)
    consensus_score: float = 0.0
    #: 收敛状态（D-82 / D-93 / D-95 / **D-96**）。三个取值对应**互不相同的补救动作**：
    #: ``approved`` 交付；``stalled`` 流程已无信息增益（不动点）；``manual_review`` 交人裁定。
    #: **「交人」内部不再细分状态值**：曾用 ``conditional``（附条件交付）与
    #: ``insufficient_evidence``（证据不足）区分交人的两种理由，但这两条信息本来就在专家意见里
    #: （有人反对 / 全部附条件 / 全员弃权），为它们各加一个状态取值只会让每个消费者都多一个
    #: 必须分支的取值。现在统一由 :data:`ReviewReason` 承载（D-96 推翻 D-93 / D-95 的取值部分）。
    consensus_status: Literal["approved", "manual_review", "stalled"] = "manual_review"
    #: 交人的**原因**（D-96），仅在 ``consensus_status == "manual_review"`` 时有值。交付与停滞
    #: 不是「交人」，所以不带原因标签——`None` 与「原因未知」必须分得开。
    review_reason: ReviewReason | None = None
    stall: StallReport = Field(default_factory=StallReport)
    #: 结构化的证据缺口清单（D-94）。此前「缺什么」只沉在渲染文本里，机器消费者拿不到；
    #: 现在它是契约的一部分：每条都可直接喂给检索端（``RetrieverPort.search()`` 落地后即刻生效）。
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)
    #: 交付物必须携带的**前置条件**（D-95 / D-96）：专家写下的「施工/采购前必须满足什么」，去重
    #: 排序后的汇总。它们与终态**无关地**跟随交付物——交付时它是交付条件，交人时它是人裁定的
    #: 依据。若只在某个状态下携带，条件就会在别的状态里消失。
    conditions: list[str] = Field(default_factory=list)
    grounding: Grounding = Field(default_factory=Grounding)
    assurance: AssuranceLevel = Field(default_factory=AssuranceLevel)
    usage: Usage = Field(default_factory=Usage)
    warnings: tuple[str, ...] = ()
    rounds: int = 0
    #: 方案产物（D-98）。**只有可交付终态才有**：`manual_review`（评审未决）走的是「回中段再跑」，
    #: 不是交付路径，所以那两份产物是 `None`。
    plan: PlanDraft | None = None
    #: 方案由谁产出：`agent`（模型撰写）或 `template`（确定性回落）。降级必须显式（P5）。
    plan_source: PlanSource | None = None
    #: 交付物文本（渲染后的方案）。人读这一份；机器读 `plan`。
    plan_markdown: str = ""


# --------------------------------------------------------------------------- #
# Orchestration (design.md §5.4 / §6.6.1 — D-70…D-91)
# --------------------------------------------------------------------------- #

AgentKind = Literal["meta", "expert", "planner"]
StepKind = Literal["think", "act", "observe", "llm", "tool", "guard"]
MetaAction = Literal[
    "read_memory",
    "dispatch_experts",
    "request_evidence",
    "attribute",
    "ask_human",
    "finalize",
]

#: 元智能体动作空间的**封闭枚举**（§5.4.1）。守卫按它校验，未知动作一律拒绝。
META_ACTIONS: tuple[str, ...] = (
    "read_memory",
    "dispatch_experts",
    "request_evidence",
    "attribute",
    "ask_human",
    "finalize",
)


class StepRecord(BaseModel):
    """位置寻址回放的最小单位（§9.4 / D-85）。

    每次 think / act / observe / llm / tool / guard 都要落一条，且**在动作执行前**
    先落盘：进程若在动作中途崩掉，重放靠它判断「这一步是否已经付过费」。因此事件
    日志里成对出现——执行前 ``step``，执行后 ``step_done``（append-only 的 JSONL
    改不了前一条，见 ``agents/runtime.py``）。
    """

    run_id: str
    node: str = ""  # "meta" / "dispatch" / "expert:E01" / "guard"
    round: int = 0
    step: int = 0  # 该节点内的步序，从 0 起
    kind: StepKind = "think"
    args_hash: str = ""  # 参数指纹（不含自由文本原文）
    produced_ids: list[str] = Field(default_factory=list)
    outcome: Literal["ok", "rejected", "failed"] = "ok"


class ActionProposal(BaseModel):
    """元智能体每一步的产出：**提案，不是执行**（D-71）。

    ``rationale`` 只进事件日志与审计，**永不进任何子智能体 prompt**（D-76）。
    """

    action: MetaAction
    payload: dict = Field(default_factory=dict)
    rationale: str = ""


class GuardVerdict(BaseModel):
    """守卫对一份提案的裁定。守卫是全局状态的**唯一写者**（§5.4.2）。"""

    action: Literal["accept", "correct", "reject"] = "accept"
    corrections: list[str] = Field(default_factory=list)
    reason: str = ""


class EvidenceRequest(BaseModel):
    """事实性检索请求（agent → 守卫）。只能在**下一轮**生效（D-16）。"""

    expert: str = "meta"
    query: str
    reason: str
    scope: Literal["components", "departments", "cases", "standards"] = "cases"
    effective_round: int = 0


class AttributionProposal(BaseModel):
    """分歧归因（§5.2.6）：重叠度由守卫**确定性**算出，LLM 只在 mixed 边界介入。"""

    round: int
    kind: Literal["judgment", "evidence", "mixed"] = "judgment"
    evidence_overlap: float = 0.0
    divergent_evidence: list[str] = Field(default_factory=list)
    rationale: str = ""


class RoundFold(BaseModel):
    """L4 过程层的**确定性**折叠（D-74）：禁 LLM 摘要，只挑结构化字段。"""

    round: int
    consensus_score: float = 0.0
    dissent_count: int = 0
    active_experts: list[str] = Field(default_factory=list)


class MemoryView(BaseModel):
    """``read_memory`` 的返回：全为枚举与结构化字段，**不含自由文本**（§5.4.11）。

    元智能体是唯一被授予全局记忆读权限的 agent（§8.4.4）；子智能体拿不到它。
    """

    view: Literal["baseline", "opinions", "rounds"] = "baseline"
    round: int = 0
    evidence_ids: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    decisions: dict[str, str] = Field(default_factory=dict)
    folds: list[RoundFold] = Field(default_factory=list)


class AgentOutcome(BaseModel):
    """``AgentRuntime`` 一次执行的产出（§12 1b 在此定稿）。"""

    agent: str
    kind: AgentKind = "expert"
    status: Literal["ok", "rejected", "abstained", "failed"] = "ok"
    steps: int = 0
    produced_ids: list[str] = Field(default_factory=list)
    note: str = ""


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