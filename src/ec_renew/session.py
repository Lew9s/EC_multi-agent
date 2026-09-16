"""Multi-turn session state.

Run is stateless, Session is the accumulator (design rule §8.4.1). A run only
ever receives an immutable ``SessionSnapshot`` — it has no reference to this
object, so it physically cannot mutate the session.

Retention rules (§8.4.6):
* the first request is kept verbatim forever as an anchor (anti-drift);
* the last ``window`` turns keep their raw request text;
* evidence is referenced by id, never copied.

The snapshot must be a **deep** copy, not merely a new tuple: a tuple holding the
same ``TurnSummary`` objects still hands the run a live handle on session state,
and then "physically cannot mutate" is a comment rather than a property.
"""

from __future__ import annotations

from .contracts import RunResult, SessionSnapshot, TurnSummary

DEFAULT_WINDOW = 3


def summarize(result: RunResult, *, turn: int) -> TurnSummary:
    """Structured turn delta for the session accumulator (§8.4.6 / §8.4.7).

    Built from **fields**, deliberately not from the rendered conclusion: that
    text opens with a constant title, so taking its first line — which is what
    the CLI used to do — stored the same string on every turn and the session
    ended up remembering nothing. No LLM is involved, so the summary cannot
    drift; that is the whole point of §8.4.7's "structured summary".

    ``open_threads`` carries the run's warning codes verbatim. They are what the
    next turn would actually have to pick up (``max_rounds_reached`` is an
    unresolved consensus, ``no_history`` an unresolved evidence gap), and they
    cross a turn boundary as machine-readable codes rather than prose.
    """
    return TurnSummary(
        turn=turn,
        user_request=result.request,
        conclusion=(
            f"{result.consensus_status}｜共识分 {result.consensus_score:.2f}"
            f"｜{result.rounds} 轮｜{result.assurance.level}"
        ),
        confirmed_experts=list(result.active_experts),
        evidence_ids=list(result.evidence_ids),
        assurance=result.assurance.model_copy(deep=True),
        open_threads=list(result.warnings),
    )


class SessionState:
    def __init__(self, session_id: str = "default", window: int = DEFAULT_WINDOW) -> None:
        self.session_id = session_id
        self.window = window
        self.pending_run_id: str | None = None
        self._anchor = ""
        self._turns: list[TurnSummary] = []
        self._constraints: list[str] = []

    # -- read --------------------------------------------------------------
    @property
    def anchor_request(self) -> str:
        return self._anchor

    @property
    def turns(self) -> list[TurnSummary]:
        return list(self._turns)

    def snapshot(self) -> SessionSnapshot:
        recent = self._turns[-self.window :] if self.window > 0 else []
        return SessionSnapshot(
            session_id=self.session_id,
            anchor_request=self._anchor,
            # deep=True: without it the snapshot hands over the very objects the
            # session stores, so a run could mutate session state in place.
            recent_turns=tuple(turn.model_copy(deep=True) for turn in recent),
            user_constraints=tuple(self._constraints),
        )

    # -- write -------------------------------------------------------------
    def add_constraint(self, text: str) -> None:
        text = text.strip()
        if text and text not in self._constraints:
            self._constraints.append(text)

    def append(self, delta: TurnSummary) -> None:
        if not self._anchor:
            self._anchor = delta.user_request  # 锚点：首轮原话，永不摘要
        self._turns.append(delta)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SessionState(id={self.session_id!r}, turns={len(self._turns)}, "
            f"pending={self.pending_run_id!r})"
        )