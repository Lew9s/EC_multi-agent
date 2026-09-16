"""Constraint tests.

Each test pins one invariant the design (docs/design.md) depends on. If one of
these starts failing, the corresponding design rule has been broken.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ec_renew.contracts import (
    DisclosurePolicy,
    ExpertOpinion,
    ExpertTask,
    RevisionContext,
)
from ec_renew.memory import EvidenceRegistry, MemoryService
from ec_renew.rag import disciplines_for_departments
from ec_renew.workflow import consensus

EID = "E-aaaaaaaaaaaa"


def opinion(expert: str, decision: str) -> ExpertOpinion:
    return ExpertOpinion(expert=expert, decision=decision, evidence_ids=[EID])


# --------------------------------------------------------------------------- #
# D-38 — consensus denominator
# --------------------------------------------------------------------------- #


def test_abstaining_cannot_inflate_consensus() -> None:
    """The defect found in CDIACR: a missing expert must never raise the score.

    The old implementation summed weights over *present* opinions only, so an
    abstaining expert was dropped from the denominator and the score went UP.
    Here the denominator is the configured weight sum, so abstaining can only
    lower the score. ``min_effective=2`` keeps the quorum gate out of the way.
    """
    weights = {"E01": 1.0, "E02": 1.0, "E03": 1.0}
    everyone_approves = {e: opinion(e, "approve") for e in weights}
    one_abstains = dict(everyone_approves)
    one_abstains["E02"] = opinion("E02", "abstain")

    full, _, _ = consensus(everyone_approves, weights, 0.6, min_effective=2)
    abstained, _, _ = consensus(one_abstains, weights, 0.6, min_effective=2)

    assert full == pytest.approx(1.0)
    assert abstained < full, "abstaining must lower the score, never raise it"
    # consensus() rounds to 4dp on purpose (reproducible serialisation).
    assert abstained == pytest.approx(2 / 3, abs=1e-4)


def test_quorum_below_min_effective_forces_manual_review() -> None:
    weights = {"E01": 1.0, "E02": 1.0, "E03": 1.0}
    opinions = {
        "E01": opinion("E01", "approve"),
        "E02": opinion("E02", "abstain"),
        "E03": opinion("E03", "abstain"),
    }
    _, effective, status = consensus(opinions, weights, 0.6, min_effective=3)
    assert effective == 1
    assert status == "manual_review"


# --------------------------------------------------------------------------- #
# D-49 / D-51 — Delphi isolation
# --------------------------------------------------------------------------- #


def test_disclosure_policy_is_derived_from_round_not_chosen() -> None:
    assert DisclosurePolicy.for_round(1) is DisclosurePolicy.NONE
    assert DisclosurePolicy.for_round(2) is DisclosurePolicy.ANONYMOUS_CLAIMS
    assert DisclosurePolicy.for_round(3) is DisclosurePolicy.ANONYMOUS_CLAIMS


def _peer_with_claim(expert: str, eid: str) -> ExpertOpinion:
    return ExpertOpinion(
        expert=expert,
        decision="reject",
        evidence_ids=[eid],
        claims=[{"claim": f"{expert} 的匿名约束", "evidence_ids": [eid], "discipline": expert}],
        constraints=[f"{expert} 的硬约束"],
    )


def test_round_one_experts_are_fully_isolated() -> None:
    """No expert may see any peer material in round 1."""
    registry = EvidenceRegistry()
    eid = registry.register(source="graph", content="case A", disciplines=["E01"]).evidence_id
    registry.freeze(1, [eid])

    task, record = MemoryService(registry).project(
        expert="E06",
        request="r",
        round_no=1,
        opinions=[_peer_with_claim("E01", eid), _peer_with_claim("E03", eid)],
    )
    assert task.cross_agent is None, "round 1 must disclose nothing"
    assert task.revision is None
    assert record.disclosed_claim_ids == []


def test_round_two_discloses_anonymous_claims_but_excludes_own() -> None:
    registry = EvidenceRegistry()
    eid = registry.register(source="graph", content="case A").evidence_id
    registry.freeze(2, [eid])
    peers = [_peer_with_claim("E01", eid), _peer_with_claim("E03", eid)]

    task, _ = MemoryService(registry).project(
        expert="E01", request="r", round_no=2, opinions=peers
    )
    assert task.cross_agent is not None
    disciplines = {c.discipline for c in task.cross_agent.anonymous_claims}
    assert "E01" not in disciplines, "an expert must not receive its own claim back"
    assert disciplines == {"E03"}
    # Hard constraints bypass filtering entirely.
    assert set(task.cross_agent.hard_constraints) == {"E01 的硬约束", "E03 的硬约束"}


def test_claim_text_is_copied_verbatim() -> None:
    """Projection must never paraphrase — that is how conditions get lost."""
    registry = EvidenceRegistry()
    eid = registry.register(source="graph", content="case A").evidence_id
    registry.freeze(2, [eid])
    original = "若设备安装先于开孔，则存在可达性冲突"

    peer = ExpertOpinion(
        expert="E02",
        decision="revise",
        evidence_ids=[eid],
        claims=[{"claim": original, "evidence_ids": [eid], "discipline": "E02"}],
    )
    task, _ = MemoryService(registry).project(
        expert="E01", request="r", round_no=2, opinions=[peer]
    )
    assert task.cross_agent.anonymous_claims[0].claim == original


def test_revision_context_carries_own_previous_and_new_evidence() -> None:
    registry = EvidenceRegistry()
    eid = registry.register(source="graph", content="case A").evidence_id
    registry.freeze(2, [eid])
    mine = opinion("E01", "revise")

    task, _ = MemoryService(registry).project(
        expert="E01", request="r", round_no=2, previous=mine, opinions=[mine]
    )
    assert task.mode == "revise"
    assert task.revision is not None
    assert task.revision.own_previous.expert == "E01"
    # An expert revising its own opinion gets no peer claims back.
    assert task.revision.feedback.anonymous_dissent == []


# --------------------------------------------------------------------------- #
# D-47 — revision ownership
# --------------------------------------------------------------------------- #


def test_an_expert_cannot_inherit_another_experts_opinion() -> None:
    mine = opinion("E01", "approve")
    with pytest.raises(ValidationError):
        ExpertTask(
            expert="E02",
            request="r",
            round=2,
            mode="revise",
            revision=RevisionContext(own_previous=mine),
        )


def test_revise_mode_requires_revision_context() -> None:
    with pytest.raises(ValidationError):
        ExpertTask(expert="E01", request="r", round=2, mode="revise")


# --------------------------------------------------------------------------- #
# Injection surface
# --------------------------------------------------------------------------- #


def test_claim_length_is_capped() -> None:
    """Bounds the injection surface when a claim is projected into a peer."""
    with pytest.raises(ValidationError):
        ExpertOpinion(
            expert="E01",
            decision="approve",
            evidence_ids=[EID],
            claims=[{"claim": "x" * 201, "evidence_ids": [EID]}],
        )


def test_opinion_requires_at_least_one_evidence_id() -> None:
    with pytest.raises(ValidationError):
        ExpertOpinion(expert="E01", decision="approve", evidence_ids=[])


def test_claim_requires_at_least_one_evidence_id() -> None:
    with pytest.raises(ValidationError):
        ExpertOpinion(
            expert="E01",
            decision="approve",
            evidence_ids=[EID],
            claims=[{"claim": "无依据的结论", "evidence_ids": []}],
        )


# --------------------------------------------------------------------------- #
# Determinism helpers
# --------------------------------------------------------------------------- #


def test_evidence_ids_are_content_addressed_and_order_independent() -> None:
    first = EvidenceRegistry()
    second = EvidenceRegistry()
    a = first.register(source="graph", content="case A")
    b = first.register(source="graph", content="case B")
    # Reversed registration order -> identical id set.
    second.register(source="graph", content="case B")
    second.register(source="graph", content="case A")
    assert sorted([a.evidence_id, b.evidence_id]) == second.all_ids()
    # Same content again -> same id, no duplicate.
    assert first.register(source="graph", content="case A").evidence_id == a.evidence_id
    assert len(first.all_ids()) == 2


def test_baseline_is_sorted_and_drops_unknown_ids() -> None:
    registry = EvidenceRegistry()
    ids = [registry.register(source="graph", content=c).evidence_id for c in ("b", "a", "c")]
    registry.freeze(1, [*ids, "E-doesnotexist"])
    assert registry.baseline(1) == sorted(ids)


def test_unknown_department_maps_to_quality() -> None:
    assert disciplines_for_departments(["船体车间"]) == ["E01"]
    assert "E03" in disciplines_for_departments(["某个不认识的部门"])
