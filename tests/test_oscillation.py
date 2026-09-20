"""§5.4.4 / D-81 — oscillation is recorded, never acted upon.

The detection predicate is deterministic; the *action* it would justify (stop
re-running the expert, or declare non-convergence) rests on a threshold that is
not calibrated yet (Q-17). So these tests pin two opposite things at once:

* the predicate is precise — a reversal on the approve > revise > reject axis,
  computed over the decision history, with ``abstain`` excluded;
* it changes nothing about the run — the expert is re-run every round, and no
  round is skipped or cut short.
"""

from __future__ import annotations

import asyncio
import json
import re

from ec_renew.agents.experts import select_experts
from ec_renew.agents.memory import EvidenceRegistry
from ec_renew.contracts import (
    EXPERT_IDS,
    ExpertOpinion,
    LLMCallMeta,
    LLMResult,
    RunInput,
    Usage,
)
from ec_renew.ports import RunContext
from ec_renew.workflow import detect_oscillations, run

EID = "E-9c4b1e7a2f03"
REQUEST = "301分段FR36污水井更换加厚板"


# --------------------------------------------------------------------------- #
# The predicate
# --------------------------------------------------------------------------- #


def _history(**by_expert: tuple[str, ...]) -> dict[int, dict[str, ExpertOpinion]]:
    """``_history(E01=("approve", "reject", "approve"))`` -> round-wise history."""
    history: dict[int, dict[str, ExpertOpinion]] = {}
    for expert, decisions in by_expert.items():
        for round_no, decision in enumerate(decisions, start=1):
            history.setdefault(round_no, {})[expert] = ExpertOpinion(
                expert=expert, decision=decision, evidence_ids=[EID]
            )
    return history


def test_a_reversal_is_detected() -> None:
    assert detect_oscillations(_history(E01=("approve", "reject", "approve"))) == {"E01": 1}


def test_a_monotone_sequence_is_not_a_reversal() -> None:
    assert detect_oscillations(_history(E01=("approve", "revise", "reject"))) == {}


def test_two_rounds_are_not_enough_to_see_a_reversal() -> None:
    """A reversal is defined on two consecutive steps, so it cannot appear before
    round 3 — worth pinning, because it bounds how early the flag can show up."""
    assert detect_oscillations(_history(E01=("approve", "reject"))) == {}


def test_every_direction_change_is_counted() -> None:
    """The counts are the whole point of recording: Q-17's threshold ("连续 2 轮
    反向 / 其它") can only be chosen from a distribution of them."""
    series = ("approve", "reject", "approve", "reject")
    assert detect_oscillations(_history(E01=series)) == {"E01": 2}


def test_abstain_neither_breaks_nor_creates_a_reversal() -> None:
    """``abstain`` is the absence of a judgment, not a point on the axis.

    Reusing ``DECISION_SCORES`` here would get this wrong: it scores ``abstain``
    the same as ``reject``, which would turn "approve -> abstain -> approve" into
    a reversal that never happened.
    """
    assert detect_oscillations(_history(E01=("approve", "abstain", "approve"))) == {}
    assert detect_oscillations(
        _history(E01=("approve", "reject", "abstain", "approve"))
    ) == {"E01": 1}


def test_only_the_experts_that_reversed_are_reported() -> None:
    history = _history(
        E01=("approve", "reject", "approve"),
        E02=("revise", "revise", "revise"),
    )
    assert detect_oscillations(history) == {"E01": 1}


def test_an_expert_added_later_is_judged_on_its_own_sequence() -> None:
    """The participating set may only grow (D-19), so a late expert's sequence
    starts late, and it must not be padded with the rounds it missed."""
    history = _history(E01=("approve", "revise", "revise", "revise"))
    for round_no, decision in ((2, "approve"), (3, "reject"), (4, "approve")):
        history[round_no]["E02"] = ExpertOpinion(
            expert="E02", decision=decision, evidence_ids=[EID]
        )

    assert detect_oscillations(history) == {"E02": 1}


# --------------------------------------------------------------------------- #
# End to end: recorded, and irrelevant to the control flow
# --------------------------------------------------------------------------- #


class _ScriptedLLM:
    """Decisions from a table, keyed by round then expert.

    The rationale carries the round, so consecutive rounds never look identical:
    otherwise the (correct) fixed-point check would end the loop early and there
    would be no oscillation left to observe.
    """

    def __init__(self, decisions: dict[int, dict[str, str]]) -> None:
        self.decisions = decisions
        self.calls: dict[str, int] = {}

    async def complete(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        meta: LLMCallMeta | None = None,
    ) -> LLMResult:
        if purpose == "meta_decision":
            # 决策层给一个「不补差集」的合法决策：本用例观察的是振荡，激活集必须
            # 完全由规则骨架决定，`calls` 也仍然只按专家计数。
            return LLMResult(
                content=json.dumps({"add_experts": [], "rationale": "scripted：不补差集"}),
                model="scripted",
                usage=Usage(calls=1),
            )
        round_no = meta.round if meta is not None else 1
        expert = meta.expert if meta is not None else "E01"
        self.calls[expert] = self.calls.get(expert, 0) + 1
        decision = self.decisions[round_no][expert]
        cited = sorted(set(re.findall(r"E-[0-9a-f]{12}", user))) or [EID]
        return LLMResult(
            content=json.dumps(
                {
                    "decision": decision,
                    "rationale": f"第 {round_no} 轮：{decision}",
                    "evidence_ids": cited,
                    "risk_level": "low",
                }
            ),
            model="scripted",
            usage=Usage(calls=1),
        )


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


#: Four rounds on purpose. A reversal only becomes visible once round 3 is in, so
#: round 4 is the **first** round on which the action §5.4.4 describes (stop
#: re-running the oscillating expert, reuse its previous opinion) could show up.
#: Under a three-round cap an interception would be indistinguishable from none.
_MAX_ROUNDS = 4
_FLIPPER_SERIES = ("approve", "reject", "approve", "reject")


def _plan_with_one_flipper(flipper: str) -> dict[int, dict[str, str]]:
    """``flipper`` reverses twice; everyone else holds "revise".

    The caller raises the threshold so no round can be approved: with a single
    approver the score is ``0.5 + 0.5/n`` (0.667 at n=3), which would otherwise
    end the loop before the reversals had a chance to appear.
    """
    rounds = range(1, _MAX_ROUNDS + 1)
    plan = {round_no: {expert: "revise" for expert in EXPERT_IDS} for round_no in rounds}
    for round_no, decision in zip(rounds, _FLIPPER_SERIES):
        plan[round_no][flipper] = decision
    return plan


def test_an_oscillating_expert_is_recorded_and_still_re_run_every_round() -> None:
    """Both halves in one place: it *is* reported, and it *changes nothing*."""
    # No retriever: the active set then depends only on the rule table, so the
    # test can predict it with the same call the workflow makes.
    active = select_experts(REQUEST, ())
    flipper = active[0]
    llm = _ScriptedLLM(_plan_with_one_flipper(flipper))
    events = _RecordingSink()
    ctx = RunContext(
        run_id="osc",
        llm=llm,
        registry=EvidenceRegistry(),
        events=events,
        retriever=None,
    )

    result = asyncio.run(
        run(RunInput(request=REQUEST), ctx, max_rounds=_MAX_ROUNDS, threshold=0.9)
    )

    # …recorded and visible…
    assert result.active_experts == active
    assert result.stall.oscillating_experts == [flipper]
    assert result.stall.reversals == {flipper: 2}
    assert not result.stall.stalled, "振荡不是不动点，两者不得混用（D-82）"
    assert f"{flipper}（2 次反向）" in result.conclusion
    assert [name for name, _ in events.events].count("oscillation_observed") == 1
    # …and not acted upon (D-81): nothing was skipped, reused or cut short.
    assert llm.calls == {expert: _MAX_ROUNDS for expert in active}, (
        "第 4 轮必须重新调用振荡专家 —— 复用上一轮意见就是 D-81 明令暂缓的那个动作"
    )
    assert result.rounds == _MAX_ROUNDS
    assert result.consensus_status == "manual_review"
    assert "max_rounds_reached" in result.warnings


def test_a_run_without_reversals_reports_nothing() -> None:
    """The absence of a finding must stay distinguishable from "not checked"."""
    active = select_experts(REQUEST, ())
    plan = {
        round_no: {expert: "revise" for expert in EXPERT_IDS}
        for round_no in range(1, _MAX_ROUNDS + 1)
    }
    llm = _ScriptedLLM(plan)
    events = _RecordingSink()
    ctx = RunContext(
        run_id="steady",
        llm=llm,
        registry=EvidenceRegistry(),
        events=events,
        retriever=None,
    )

    result = asyncio.run(
        run(RunInput(request=REQUEST), ctx, max_rounds=_MAX_ROUNDS, threshold=0.9)
    )

    assert result.active_experts == active
    assert result.stall.oscillating_experts == []
    assert result.stall.reversals == {}
    assert "迭代振荡" not in result.conclusion
    assert not any(name == "oscillation_observed" for name, _ in events.events)
