"""§8.4 — the session accumulator and the snapshot seam.

The session is the only channel across conversation turns (D-43), so its two hard
rules have to be properties rather than comments:

* a run cannot mutate session state through the snapshot it is handed (§8.4.1);
* no session history may reach a sub-agent prompt (§8.4.5).
"""

from __future__ import annotations

import asyncio
import json
import re

from ec_renew.agents.memory import EvidenceRegistry
from ec_renew.contracts import (
    AssuranceLevel,
    LLMCallMeta,
    LLMResult,
    RunInput,
    RunResult,
    TurnSummary,
    Usage,
)
from ec_renew.observability import NullEventLog
from ec_renew.ports import RunContext
from ec_renew.rag import InMemoryRetriever
from ec_renew.session import SessionState, summarize
from ec_renew.workflow import run


def _result(
    *,
    request: str = "请求",
    status: str = "approved",
    score: float = 0.75,
    rounds: int = 2,
    level: str = "history_backed",
    warnings: tuple[str, ...] = (),
) -> RunResult:
    return RunResult(
        request=request,
        normalized_request=request,
        # The rendered report always opens with this constant title, so a
        # summary derived from it would store nothing per-turn.
        conclusion="# 工程变更方案\n\n- 状态：approved\n",
        consensus_status=status,  # type: ignore[arg-type]
        consensus_score=score,
        rounds=rounds,
        assurance=AssuranceLevel(level=level),  # type: ignore[arg-type]
        warnings=warnings,
        active_experts=["E01", "E03"],
        evidence_ids=["E-9c4b1e7a2f03"],
    )


# --------------------------------------------------------------------------- #
# §8.4.6 / §8.4.7 — what the accumulator stores
# --------------------------------------------------------------------------- #


def test_stored_conclusion_comes_from_fields_not_from_the_rendered_title() -> None:
    """``result.conclusion.splitlines()[0]`` is the constant ``# 工程变更方案``.

    A stored summary derived from it would be the same on *every* turn, and the
    accumulator would remember nothing.
    """
    result = _result(status="stalled", score=0.5, rounds=3, warnings=("stalled",))

    summary = summarize(result, turn=4)

    assert "# 工程变更方案" not in summary.conclusion
    assert "stalled" in summary.conclusion
    assert "0.50" in summary.conclusion
    assert "3 轮" in summary.conclusion
    assert summary.turn == 4
    assert summary.confirmed_experts == ["E01", "E03"]
    assert summary.evidence_ids == ["E-9c4b1e7a2f03"]
    assert summary.open_threads == ["stalled"]


def test_different_outcomes_store_different_conclusions() -> None:
    """Different outcomes must store different conclusions — otherwise the
    accumulator cannot tell two turns apart.

    (Two runs *may* legitimately agree — but then they agree on the fields, not
    because the stored text is a constant.)
    """
    first = summarize(_result(status="approved", score=0.75), turn=1)
    second = summarize(_result(status="manual_review", score=0.42), turn=2)

    assert first.conclusion != second.conclusion


def test_snapshot_keeps_the_anchor_and_only_the_last_window_turns() -> None:
    session = SessionState(window=2)
    for turn in range(1, 5):
        session.append(
            TurnSummary(turn=turn, user_request=f"请求{turn}", conclusion=f"结论{turn}")
        )

    snapshot = session.snapshot()

    assert snapshot.anchor_request == "请求1", "锚点 = 首轮原话，永不摘要（§8.4.7）"
    assert [turn.turn for turn in snapshot.recent_turns] == [3, 4]


def test_the_turn_summary_field_set_is_frozen() -> None:
    """§8.4.6 decides what may cross a conversation turn; widening it — say, by
    adding per-expert opinions — is a design change, so it must be deliberate."""
    assert set(TurnSummary.model_fields) == {
        "turn",
        "user_request",
        "conclusion",
        "confirmed_experts",
        "evidence_ids",
        "assurance",
        "open_threads",
    }


# --------------------------------------------------------------------------- #
# §8.4.1 — the run gets a copy, not a handle on session state
# --------------------------------------------------------------------------- #


def test_a_run_cannot_mutate_the_session_through_its_snapshot() -> None:
    """A tuple of the *same* ``TurnSummary`` objects still exposes session state."""
    session = SessionState()
    session.append(TurnSummary(turn=1, user_request="原始请求", conclusion="原始结论"))

    snapshot = session.snapshot()
    snapshot.recent_turns[0].conclusion = "被 run 改掉了"
    snapshot.recent_turns[0].user_request = "也被改了"

    assert session.turns[0].conclusion == "原始结论"
    assert session.turns[0].user_request == "原始请求"


# --------------------------------------------------------------------------- #
# §8.4.5 — nothing from the session reaches a sub-agent
# --------------------------------------------------------------------------- #

_SENTINEL = "绝不应进prompt"


class _RecordingLLM:
    """Returns valid opinions, and keeps every prompt it was shown."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.purposes: list[str] = []

    async def complete(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        meta: LLMCallMeta | None = None,
    ) -> LLMResult:
        self.prompts.append(user)
        self.purposes.append(purpose)
        cited = sorted(set(re.findall(r"E-[0-9a-f]{12}", user))) or ["E-9c4b1e7a2f03"]
        return LLMResult(
            content=json.dumps(
                {
                    "decision": "approve",
                    "rationale": "r",
                    "evidence_ids": cited,
                    "risk_level": "low",
                }
            ),
            model="recording",
            usage=Usage(calls=1),
        )


def test_no_session_history_reaches_a_sub_agent_prompt() -> None:
    """§8.4.5: the session's only reader is the meta agent.

    A sub-agent gets this round's request and its own evidence — never the
    anchor, the previous conclusions, the open threads or the user constraints.
    The sentinel makes that unambiguous: if it ever appears in a prompt, some
    change has opened the channel this rule exists to close.
    """
    llm = _RecordingLLM()
    session = SessionState()
    session.append(
        TurnSummary(
            turn=1,
            user_request=f"锚点请求-{_SENTINEL}",
            conclusion=f"上一轮结论-{_SENTINEL}",
            open_threads=[f"未决事项-{_SENTINEL}"],
        )
    )
    session.add_constraint(f"用户约束-{_SENTINEL}")
    ctx = RunContext(
        run_id="session-guard",
        llm=llm,
        registry=EvidenceRegistry(),
        events=NullEventLog(),
        retriever=InMemoryRetriever(),
    )

    asyncio.run(
        run(
            RunInput(request="301分段FR36污水井更换加厚板", session=session.snapshot()),
            ctx,
        )
    )

    assert llm.prompts, "没捕获到任何 prompt，这条用例就失去意义了"
    # 会话历史不进**评审专家**的 prompt（独立判断不能被上一轮污染）。方案撰写者（D-98）是例外：
    # 它按设计要拿到会话上下文与用户意见——那里的偏离由 D-100 记录，与这条断言不冲突。
    expert_prompts = [
        prompt for prompt, purpose in zip(llm.prompts, llm.purposes) if purpose == "expert"
    ]
    assert expert_prompts, "前提：本轮确实派发过专家"
    for prompt in expert_prompts:
        assert _SENTINEL not in prompt
