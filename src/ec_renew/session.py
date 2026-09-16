"""Multi-turn session state.

Run is stateless, Session is the accumulator (design rule §8.4.1). A run only
ever receives an immutable ``SessionSnapshot`` — it has no reference to this
object, so it physically cannot mutate the session.

Retention rules (§8.4.6):
* the first request is kept verbatim forever as an anchor (anti-drift);
* the last ``window`` turns keep their raw request text;
* evidence is referenced by id, never copied.
"""

from __future__ import annotations

from .contracts import SessionSnapshot, TurnSummary

DEFAULT_WINDOW = 3


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
            recent_turns=tuple(recent),
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