"""Run-scoped memory: the evidence registry, and deterministic projection.

Design rules enforced here (see docs/design.md §5.4.6 / §8.5):

* ``EvidenceRegistry`` is content-addressed, so writes from parallel branches
  are order-independent and replayed runs get identical ids.
* ``MemoryService.project`` is a **deterministic function**, never an LLM tool.
  ``claim`` text is copied verbatim; nothing is re-summarised.
* Hard constraints bypass filtering and always reach every expert.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from ..contracts import (
    AgentBudget,
    AnonymizedClaim,
    CrossAgentInfo,
    DisclosurePolicy,
    Evidence,
    EvidenceRef,
    ExpertOpinion,
    ExpertTask,
    MemoryView,
    ProjectionRecord,
    ReviewFeedback,
    RevisionContext,
    RoundFold,
    SubQuestion,
)
from ..errors import InvariantViolation


def make_evidence_id(source: str, content: str) -> str:
    """Content-addressed id: same fact -> same id, regardless of arrival order."""
    digest = hashlib.sha1(f"{source}|{content}".encode()).hexdigest()[:12]
    return f"E-{digest}"


def _dedupe_sorted(values: Iterable[str]) -> list[str]:
    return sorted({v for v in values if v})


class EvidenceRegistry:
    """The single source of truth for facts within one run."""

    def __init__(self) -> None:
        self._store: dict[str, Evidence] = {}
        self._baselines: dict[int, list[str]] = {}

    # -- write -------------------------------------------------------------
    def register(
        self,
        *,
        source: str,
        content: str,
        entity_keys: Iterable[str] = (),
        disciplines: Iterable[str] = (),
        group_keys: Iterable[str] = (),
        source_file: str = "",
        score: float = 0.0,
    ) -> Evidence:
        eid = make_evidence_id(source, content)
        existing = self._store.get(eid)
        if existing is not None:
            # First write wins -> replay deterministic even if scores differ.
            existing.score = max(existing.score, score)
            return existing

        evidence = Evidence(
            evidence_id=eid,
            source=source,  # type: ignore[arg-type]
            content=content,
            entity_keys=_dedupe_sorted(entity_keys),
            disciplines=_dedupe_sorted(disciplines),
            group_keys=_dedupe_sorted(group_keys),
            source_file=source_file,
            score=score,
        )
        self._store[eid] = evidence
        return evidence

    def freeze(self, round_no: int, ids: Iterable[str]) -> None:
        """Freeze the baseline for a round. Sorted -> order independent."""
        known = {eid for eid in ids if eid in self._store}
        self._baselines[round_no] = sorted(known)

    # -- read --------------------------------------------------------------
    def get(self, evidence_id: str) -> Evidence | None:
        return self._store.get(evidence_id)

    def by_ids(self, ids: Iterable[str]) -> list[Evidence]:
        return [self._store[eid] for eid in ids if eid in self._store]

    def refs(self, ids: Iterable[str]) -> list[EvidenceRef]:
        out: list[EvidenceRef] = []
        for eid in ids:
            ev = self._store.get(eid)
            if ev is None:
                continue
            gist = ev.content.strip().replace("\n", " ")
            out.append(
                EvidenceRef(
                    evidence_id=eid,
                    gist=gist[:120] + ("…" if len(gist) > 120 else ""),
                    source=ev.source,
                )
            )
        return out

    def baseline(self, round_no: int) -> list[str]:
        return list(self._baselines.get(round_no, []))

    def all_ids(self) -> list[str]:
        return sorted(self._store)

    def count_by_source(self, source: str) -> int:
        return sum(1 for ev in self._store.values() if ev.source == source)


@dataclass(frozen=True)
class MemorySlice:
    """A read-only view of global memory. Kept shaped like a future tool."""

    baseline_ids: tuple[str, ...]
    conflicts: tuple[AnonymizedClaim, ...]
    hard_constraints: tuple[str, ...]


def collect_hard_constraints(opinions: Iterable[ExpertOpinion]) -> list[str]:
    """Hard constraints are never filtered away (design rule §8.5.4)."""
    return _dedupe_sorted(c for op in opinions for c in op.constraints)


def collect_claims(
    opinions: Iterable[ExpertOpinion], *, exclude: str | None = None
) -> list[AnonymizedClaim]:
    """Cross-agent claims: deterministic order, verbatim text, **no identity**.

    Returns ``AnonymizedClaim`` rather than ``Claim`` so that "a peer's
    discipline never crosses the boundary" is a property of the type, not of
    how carefully a renderer is written (D-77).
    """
    claims = [
        AnonymizedClaim.from_claim(c)
        for op in opinions
        for c in op.claims
        if op.expert != exclude
    ]
    return sorted(claims, key=lambda c: (c.claim, c.condition or ""))


class MemoryService:
    """Read / Filter / Project. Deterministic by construction."""

    def __init__(self, registry: EvidenceRegistry) -> None:
        self._reg = registry

    def read(self, round_no: int, opinions: Iterable[ExpertOpinion] = ()) -> MemorySlice:
        opinions = list(opinions)
        return MemorySlice(
            baseline_ids=tuple(self._reg.baseline(round_no)),
            conflicts=tuple(collect_claims(opinions)),
            hard_constraints=tuple(collect_hard_constraints(opinions)),
        )

    def read_view(
        self,
        view: str,
        *,
        round_no: int,
        opinions: Iterable[ExpertOpinion] = (),
        folds: Iterable[RoundFold] = (),
    ) -> MemoryView:
        """``read_memory`` 的枚举视图（§5.4.11 / D-91）：参数与返回**都不含自由文本**。

        与 ``read()`` 的分工：``read()`` 是外环内部用的 ``MemorySlice``；本方法是
        元智能体的 observe 入口，返回可枚举、可机检的三个视图——``baseline``（L3
        证据 id 与计数）、``opinions``（本轮各专家的**决策**，不含论证原文）、
        ``rounds``（L4 的确定性折叠）。明细按需取，绝不做 LLM 摘要（D-74）。
        """
        opinions = list(opinions)
        if view == "baseline":
            ids = self._reg.baseline(round_no)
            return MemoryView(
                view="baseline",
                round=round_no,
                evidence_ids=ids,
                counts={"baseline": len(ids), "registry": len(self._reg.all_ids())},
            )
        if view == "opinions":
            return MemoryView(
                view="opinions",
                round=round_no,
                decisions={
                    op.expert: op.decision for op in sorted(opinions, key=lambda o: o.expert)
                },
                counts={
                    "opinions": len(opinions),
                    "abstain": sum(1 for op in opinions if op.decision == "abstain"),
                    "claims": sum(len(op.claims) for op in opinions),
                },
            )
        if view == "rounds":
            return MemoryView(view="rounds", round=round_no, folds=list(folds))
        # 守卫在 judge() 里已经按 VALID_VIEWS 拦过一道；走到这里说明两处清单不一致。
        raise InvariantViolation(f"未知的 read_memory 视图：{view!r}")

    def project(
        self,
        *,
        expert: str,
        request: str,
        round_no: int,
        sub_questions: list[SubQuestion] | None = None,
        previous: ExpertOpinion | None = None,
        opinions: Iterable[ExpertOpinion] = (),
        consensus_score: float = 0.0,
        dissent_count: int = 0,
        budget: AgentBudget | None = None,
    ) -> tuple[ExpertTask, ProjectionRecord]:
        policy = DisclosurePolicy.for_round(round_no)
        opinions = list(opinions)
        baseline_ids = self._reg.baseline(round_no)

        cross: CrossAgentInfo | None = None
        disclosed_ids: list[str] = []
        if policy is DisclosurePolicy.ANONYMOUS_CLAIMS:
            visible = collect_claims(opinions, exclude=expert)
            # Ids, not text: the disclosure graph (§8.5.6) is only computable if
            # a claim has an identity (D-78). Deduplicated, because two experts
            # independently raising the same argument legitimately share an id.
            disclosed_ids = sorted({c.claim_id for c in visible})
            cross = CrossAgentInfo(
                anonymous_claims=visible,
                # Hard constraints bypass Filter entirely.
                hard_constraints=collect_hard_constraints(opinions),
                consensus_score=consensus_score,
                dissent_count=dissent_count,
            )

        revision: RevisionContext | None = None
        if previous is not None:
            # 「本轮比上一轮多了哪几条证据」由两次冻结基线相减**确定性**得出（D-97）。缺口回填
            # 现在真的会中途扩张基线，专家必须知道哪几条是新的——否则第 2 轮只是把同一批证据
            # 再喂一遍，而修正意见却要为此负责（D-81 的「无信息增益」也正需要这个差集来描述）。
            new_ids = sorted(set(baseline_ids) - set(self._reg.baseline(round_no - 1)))
            revision = RevisionContext(
                own_previous=previous,
                new_evidence_ids=new_ids,
                feedback=ReviewFeedback(
                    round=round_no,
                    consensus_score=consensus_score,
                    dissent_count=dissent_count,
                    anonymous_dissent_ids=disclosed_ids,
                ),
            )

        task = ExpertTask(
            mode="revise" if previous is not None else "initial",
            expert=expert,
            request=request,
            sub_questions=sub_questions or [],
            evidence_view={"round": round_no, "baseline_ids": baseline_ids},
            evidence=self._reg.refs(baseline_ids),
            autonomy="L0",
            budget=budget or AgentBudget(),
            round=round_no,
            revision=revision,
            cross_agent=cross,
        )
        record = ProjectionRecord(
            round=round_no,
            expert=expert,
            policy=policy,
            disclosed_claim_ids=disclosed_ids,
            hard_constraints=sorted(cross.hard_constraints) if cross else [],
        )
        return task, record