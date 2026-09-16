"""§12 1f — machine-readable call context travels **out of band**.

Lifting ``[[CTX …]]`` out of the prompt is only safe if the context still reaches
the two places that need it: the cache key and the fake adapter. These tests pin
both, plus the property the fixed-point argument actually rests on — that the
round counter no longer changes the prompt at all.
"""

from __future__ import annotations

import asyncio
import json

from pydantic import SecretStr

from ec_renew.agents.skills.expert_review import render_task, run_expert
from ec_renew.config import settings as base_settings
from ec_renew.contracts import ExpertTask, LLMCallMeta, LLMResult, Usage
from ec_renew.llm import DeepSeekLLM, FakeLLM, LLMCache

#: A plausible evidence id — deliberately **not** a run of sequential digits.
#: The secret scanner's exact-value judgement compares against the *local* .env,
#: whose legacy NEO4J_PASSWORD is eight digits long; a sequential literal
#: contains it verbatim and turns the secret gate red on a test fixture.
EID = "E-9c4b1e7a2f03"


# --------------------------------------------------------------------------- #
# The producer — the prompt no longer carries the header
# --------------------------------------------------------------------------- #


def test_the_round_counter_no_longer_changes_the_prompt() -> None:
    """Two tasks differing *only* in ``round`` must render byte-identically.

    This is the whole basis of §5.4.4's fixed-point argument: once the frozen
    baseline and every projected field are unchanged, the next prompt has
    nothing left to differ by. (The real round 1 → 2 transition also changes
    ``mode`` / ``revision`` / ``cross_agent`` structurally — which is exactly why
    the stall check only applies from round 2 onward.)
    """
    first = ExpertTask(expert="E01", request="r", round=1)
    second = ExpertTask(expert="E01", request="r", round=2)

    assert render_task(first) == render_task(second)


def test_the_prompt_contains_no_ctx_header() -> None:
    """Guards against the header being reintroduced "just for the fake adapter"."""
    prompt = render_task(ExpertTask(expert="E01", request="r", round=1))

    assert "[[CTX" not in prompt
    assert "expert=E01" not in prompt


class _RecordingLLM:
    """Records the out-of-band meta it is handed, and returns a valid opinion."""

    def __init__(self) -> None:
        self.metas: list[LLMCallMeta | None] = []

    async def complete(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        meta: LLMCallMeta | None = None,
    ) -> LLMResult:
        self.metas.append(meta)
        return LLMResult(
            content=json.dumps(
                {"decision": "approve", "evidence_ids": [EID], "risk_level": "low"}
            ),
            model="recording",
            usage=Usage(calls=1),
        )


def test_run_expert_hands_its_context_over_out_of_band() -> None:
    """The expert runner is the only producer of this meta in the pipeline."""
    llm = _RecordingLLM()
    task = ExpertTask(expert="E06", request="r", round=2, mode="initial")

    asyncio.run(run_expert(task, llm, [EID]))

    assert llm.metas == [LLMCallMeta(expert="E06", round=2, mode="initial")]


# --------------------------------------------------------------------------- #
# The consumers — the cache key and the fake adapter
# --------------------------------------------------------------------------- #


class _MemoryCache(LLMCache):
    """In-memory stand-in for the on-disk cache.

    Deliberately not a tmp dir: this sandbox makes ``%TEMP%`` unwritable, and the
    cache's file layout is not what this test is about — the *key* is.
    """

    def __init__(self) -> None:
        super().__init__(".cache/llm", enabled=False)
        self.store: dict[str, LLMResult] = {}

    def get(self, key: str) -> LLMResult | None:
        return self.store.get(key)

    def put(self, key: str, result: LLMResult) -> None:
        self.store[key] = result


def test_call_meta_is_part_of_the_cache_key() -> None:
    """Same prompt text, different round ⇒ two calls, not one cached answer.

    This is the trap the move creates. Round / expert / mode used to sit inside
    ``user``, so they were cache-key material for free; lifting them out of the
    message without folding them back into the key would let round 2 hit round
    1's cached opinion for the same expert — the consensus loop would silently
    degenerate, and ``stalled`` would fire on the wrong grounds.
    """
    cfg = base_settings.model_copy(update={"deepseek_api_key": SecretStr("test-key")})
    cache = _MemoryCache()
    llm = DeepSeekLLM(cfg=cfg, cache=cache)

    posted: list[dict] = []

    async def fake_post(body: dict) -> LLMResult:
        posted.append(body)
        return LLMResult(content="{}", model="stub", usage=Usage(calls=1))

    llm._post = fake_post  # type: ignore[method-assign]

    async def two_rounds() -> None:
        for round_no, mode in ((1, "initial"), (2, "revise")):
            await llm.complete(
                purpose="expert",
                system="s",
                user="u",
                meta=LLMCallMeta(expert="E01", round=round_no, mode=mode),
            )

    asyncio.run(two_rounds())

    assert len(posted) == 2, "第 2 轮命中了第 1 轮的缓存：meta 没有进缓存键"


def test_call_meta_never_reaches_the_provider() -> None:
    """It is our bookkeeping: putting it in the request body would leak the
    harness's internal structure into the prompt by the back door."""
    cfg = base_settings.model_copy(update={"deepseek_api_key": SecretStr("test-key")})
    llm = DeepSeekLLM(cfg=cfg, cache=_MemoryCache())

    bodies: list[dict] = []

    async def fake_post(body: dict) -> LLMResult:
        bodies.append(body)
        return LLMResult(content="{}", model="stub", usage=Usage(calls=1))

    llm._post = fake_post  # type: ignore[method-assign]

    asyncio.run(
        llm.complete(
            purpose="expert",
            system="s",
            user="u",
            meta=LLMCallMeta(expert="E01", round=3, mode="revise"),
        )
    )

    sent = json.dumps(bodies[0], ensure_ascii=False)
    assert sent.count("E01") == 0 and "revise" not in sent


def test_the_fake_adapter_reads_its_context_from_meta() -> None:
    """``plan`` is keyed by round and expert, so the fake must see the meta —
    this is the "FakeLLM 的驱动方式" cost §12 1f predicted."""
    llm = FakeLLM(plan={1: {"E01": "approve"}, 2: {"E01": "reject"}})

    async def decide(round_no: int) -> str:
        result = await llm.complete(
            purpose="expert",
            system="s",
            user=f"## 本轮证据\n- {EID}: case A",
            meta=LLMCallMeta(expert="E01", round=round_no, mode="initial"),
        )
        return json.loads(result.content)["decision"]

    assert asyncio.run(decide(1)) == "approve"
    assert asyncio.run(decide(2)) == "reject"
