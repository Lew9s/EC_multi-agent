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

from .contracts import (
    AgentBudget,
    Claim,
    CrossAgentInfo,
    DisclosurePolicy,
    Evidence,
    EvidenceRef,
    ExpertOpinion,
    ExpertTask,
    ProjectionRecord,
    ReviewFeedback,
    RevisionContext,
    SubQuestion,
)


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
    conflicts: tuple[Claim, ...]
    hard_constraints: tuple[str, ...]


def collect_hard_constraints(opinions: Iterable[ExpertOpinion]) -> list[str]:
    """Hard constraints are never filtered away (design rule §8.5.4)."""
    return _dedupe_sorted(c for op in opinions for c in op.constraints)


def collect_claims(
    opinions: Iterable[ExpertOpinion], *, exclude: str | None = None
) -> list[Claim]:
    """Cross-agent claims, deterministic order, verbatim text."""
    claims = [c for op in opinions for c in op.claims if op.expert != exclude]
    return sorted(claims, key=lambda c: (c.discipline, c.claim))


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
        disclosed: list[str] = []
        if policy is DisclosurePolicy.ANONYMOUS_CLAIMS:
            visible = collect_claims(opinions, exclude=expert)
            disclosed = [c.claim for c in visible]
            cross = CrossAgentInfo(
                anonymous_claims=visible,
                # Hard constraints bypass Filter entirely.
                hard_constraints=collect_hard_constraints(opinions),
                consensus_score=consensus_score,
                dissent_count=dissent_count,
            )

        revision: RevisionContext | None = None
        if previous is not None:
            revision = RevisionContext(
                own_previous=previous,
                new_evidence_ids=[],  # demo: baseline does not expand mid-run
                feedback=ReviewFeedback(
                    round=round_no,
                    consensus_score=consensus_score,
                    dissent_count=dissent_count,
                    anonymous_dissent=sorted(disclosed),
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
            disclosed_claim_ids=sorted(disclosed),
            hard_constraint_ids=sorted(cross.hard_constraints) if cross else [],
        )
        return task, record