"""Ports — the only interfaces business code is allowed to depend on.

Exception translation happens in the adapters behind these ports: code above
this layer must never see ``httpx.*`` or ``neo4j.*`` exceptions.

This is also the seam that makes the provider swappable (DeepSeek today, a
local Ollama profile later) without touching workflow or agents.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .contracts import (
    EntityRef,
    Evidence,
    EvidenceBundle,
    EvidenceRef,
    GraphExpansion,
    HumanDecision,
    HumanReviewRequest,
    LLMResult,
    ProjectionRecord,
)


@runtime_checkable
class EventSinkPort(Protocol):
    def emit(self, event: str, **fields: object) -> None: ...


@runtime_checkable
class LLMPort(Protocol):
    """One completion call.

    ``purpose`` labels the call (``intent`` / ``expert`` / ``finalize``) so the
    fake adapter can dispatch on it, and so cache keys and event logs stay
    readable.
    """

    async def complete(self, *, purpose: str, system: str, user: str) -> LLMResult: ...


@runtime_checkable
class RetrieverPort(Protocol):
    """Graph access, all async (the neo4j driver is sync, so the adapter
    runs it in a worker thread).

    ``link``      — ground the request text onto real graph entities.
    ``expand``    — graph-grounded completion (parent structure, historical
                    departments -> disciplines). This is what makes abstract
                    user requests resolvable.
    ``prefetch``  — the baseline evidence for one round.
    """

    async def link(self, request: str) -> list[EntityRef]: ...

    async def expand(self, entities: list[EntityRef]) -> GraphExpansion: ...

    async def prefetch(self, queries: list[str], top_k: int) -> EvidenceBundle: ...


@runtime_checkable
class EvidenceRegistryPort(Protocol):
    """Run-scoped store. Content-addressed, so writes are order-independent."""

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
    ) -> Evidence: ...

    def get(self, evidence_id: str) -> Evidence | None: ...

    def refs(self, ids: Iterable[str]) -> list[EvidenceRef]: ...

    def freeze(self, round_no: int, ids: Iterable[str]) -> None: ...

    def baseline(self, round_no: int) -> list[str]: ...

    def all_ids(self) -> list[str]: ...


@dataclass
class RunContext:
    """Per-run runtime context, passed explicitly to every stage.

    ``RunContext`` never carries the API key — only the LLM port does.
    """

    run_id: str
    llm: LLMPort
    registry: EvidenceRegistryPort
    events: EventSinkPort
    retriever: RetrieverPort | None = None
    projections: list[ProjectionRecord] = field(default_factory=list)
    # HITL seam: set by the CLI. `None` means run headless (warn and continue).
    ask_human: Callable[[HumanReviewRequest], Awaitable[HumanDecision]] | None = None