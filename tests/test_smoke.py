"""Offline smoke test: full pipeline, no network, no API key.

This is the demo's acceptance test (DoD items 1, 3, 4, 5).

Notes
-----
The log directory is created by the test itself rather than via pytest's
``tmp_path`` fixture, because this lab machine's sandbox denies pytest access
to its own temp directory.
"""

from __future__ import annotations

import asyncio
import itertools
from pathlib import Path

from ec_renew.contracts import DisclosurePolicy, EvidenceBundle, RunInput
from ec_renew.llm import FakeLLM
from ec_renew.memory import EvidenceRegistry
from ec_renew.observability import JsonlEventLog
from ec_renew.ports import RunContext
from ec_renew.rag import InMemoryRetriever
from ec_renew.workflow import run

LOG_DIR = Path(".pytest_run")
_counter = itertools.count(1)

# Round 1 fails (score 0.5 < 0.6); round 2 agrees -> the two-round Delphi loop
# is genuinely exercised rather than short-circuited.
PLAN = {
    1: {"E01": "revise", "E03": "reject", "E06": "approve"},
    2: {"E01": "approve", "E03": "approve", "E06": "approve"},
}


def _context(llm, retriever) -> RunContext:
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"events-{next(_counter)}.jsonl"
    return RunContext(
        run_id=f"smoke-{log_path.stem}",
        llm=llm,
        registry=EvidenceRegistry(),
        events=JsonlEventLog(log_path, log_path.stem),
        retriever=retriever,
    )


def _request() -> RunInput:
    return RunInput(request="301分段FR36污水井更换加厚板，需焊接")


def test_pipeline_runs_offline() -> None:
    ctx = _context(FakeLLM(PLAN), InMemoryRetriever())
    result = asyncio.run(run(_request(), ctx))

    # DoD 1 — a plan with evidence references comes out.
    assert result.consensus_status == "approved"
    assert result.rounds == 2
    assert result.evidence_ids
    assert "E-" in result.conclusion
    assert result.active_experts == ["E01", "E03", "E06"]
    assert result.grounding.has_history
    assert result.usage.calls >= 6

    # DoD 3 — every stage left a trace.
    log_path = Path(ctx.events.path)
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 10
    assert any('"event": "round_finished"' in line for line in lines)
    assert any('"event": "baseline_frozen"' in line for line in lines)

    # Delphi: disclosure advanced with the round.
    policies = [p.policy for p in ctx.projections]
    assert DisclosurePolicy.NONE in policies
    assert DisclosurePolicy.ANONYMOUS_CLAIMS in policies


def test_replay_is_deterministic() -> None:
    """DoD 2 — same input, byte-identical conclusion."""
    first = asyncio.run(run(_request(), _context(FakeLLM(PLAN), InMemoryRetriever())))
    second = asyncio.run(run(_request(), _context(FakeLLM(PLAN), InMemoryRetriever())))
    assert first.conclusion == second.conclusion
    assert first.evidence_ids == second.evidence_ids
    assert first.consensus_score == second.consensus_score


def test_no_history_degrades_to_knowledge_based() -> None:
    class EmptyRetriever(InMemoryRetriever):
        async def prefetch(self, queries, top_k):  # type: ignore[override]
            return EvidenceBundle(round=0, items=[], warnings=["low_evidence"])

    ctx = _context(
        FakeLLM({1: {"E01": "approve", "E03": "approve", "E06": "approve"}}),
        EmptyRetriever(),
    )
    result = asyncio.run(run(RunInput(request="换个加厚板"), ctx))

    assert result.grounding.has_history is False
    assert result.assurance.level == "knowledge_based"
    assert "no_history" in result.warnings
    assert result.consensus_status == "approved"  # still produces a plan