"""Exception hierarchy.

The governing rule (docs/design.md §7.1): **expected business outcomes are
states, not exceptions.** "consensus not reached", "insufficient evidence",
"expert abstained" are all state fields. Only the unexpected raises.

Translation from third-party exceptions happens exactly once, in the adapters
behind ``ports``. Code above that layer must never see ``httpx.*`` or
``neo4j.*`` exceptions.
"""

from __future__ import annotations


class ECError(Exception):
    """Base class for every error this system raises deliberately."""


# ---- retryable ---------------------------------------------------------- #
class TransientError(ECError):
    """Timeout / connection reset / rate limit / pool exhaustion."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimited(TransientError):
    """HTTP 429. ``retry_after`` is taken from the Retry-After header if present."""


# ---- not retryable, but degradable -------------------------------------- #
class ContractViolation(ECError):
    """LLM output did not satisfy the target schema."""

    def __init__(self, message: str, *, node: str = "", attempt: int = 1) -> None:
        super().__init__(message)
        self.node = node
        self.attempt = attempt


# ---- not retryable, hard failure ---------------------------------------- #
class PermanentExternalError(ECError):
    """Auth failure / missing index / insufficient permission."""


class InvalidRequest(ECError):
    """Empty question, unknown component name, ..."""


# ---- internal bug: fail fast, never catch ------------------------------- #
class InvariantViolation(ECError):
    """Reducer broke associativity, illegal state transition, ..."""


class BudgetExceeded(ECError):
    pass


class StepLimitExceeded(ECError):
    pass