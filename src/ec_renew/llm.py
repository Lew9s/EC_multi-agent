"""LLM adapters.

* ``FakeLLM``      — offline, deterministic, no API key. Used to prove the
                     pipeline end-to-end before spending money.
* ``DeepSeekLLM``  — OpenAI-compatible endpoint over ``httpx`` (the ``openai``
                     SDK is not installed on this lab machine). The extra body
                     field ``{"thinking": {"type": ...}}`` is passed verbatim.

Both go through ``LLMCache``: content-addressed, so re-running an experiment is
free and byte-reproducible.

Machine-readable call context (expert / round / mode) reaches both adapters as
``LLMCallMeta`` — delivered **out of band** rather than embedded in the prompt
(design §12 1f). It is part of the cache key, so moving it out of the message
text cannot silently collapse two different rounds onto one cached answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from .config import Settings
from .config import settings as default_settings
from .contracts import LLMCallMeta, LLMResult, Usage
from .errors import PermanentExternalError, RateLimited, TransientError

# --------------------------------------------------------------------------- #
# Evidence id scanning
# --------------------------------------------------------------------------- #

#: ``E-`` + 12 hex chars — the evidence ids the fake adapter cites.
#:
#: 调用上下文不进 prompt：expert / round / mode 一律以 ``LLMCallMeta`` 带外传递
#: （design §12 1f），因此本模块没有 prompt 头解析器。
_EID_RE = re.compile(r"E-[0-9a-f]{12}")


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


class LLMCache:
    """Content-addressed response cache.

    Key excludes timestamps and run ids so it is shared across runs — which is
    what makes an experiment reproducible even though the provider's model
    pointer can move underneath us.
    """

    def __init__(self, directory: Path | str, enabled: bool = True) -> None:
        self.dir = Path(directory)
        self.enabled = enabled
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(
        model: str,
        system: str,
        user: str,
        params: dict[str, Any],
        meta: LLMCallMeta | None = None,
    ) -> str:
        """Content-addressed key for one call.

        ``meta`` 是键的一部分，**这是有意的**：expert / round / mode 决定了「同一个
        prompt 文本问的是不是同一个问题」。不把它们折进键，第 2 轮就会命中第 1 轮
        同一专家的缓存答案——文本相同、问题不同。见
        ``test_call_meta_is_part_of_the_cache_key``。
        """
        payload = json.dumps(
            {
                "model": model,
                "system": system,
                "user": user,
                "params": params,
                "meta": meta.model_dump() if meta is not None else None,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> LLMResult | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return LLMResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            # 缓存文件损坏 / 读不到 —— 按未命中处理即可，重算一次比抛错划算。
            # 刻意不写裸 except：那会把「代码写错」也伪装成「缓存没命中」。
            return None

    def put(self, key: str, result: LLMResult) -> None:
        if not self.enabled:
            return
        self._path(key).write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
# Fake adapter
# --------------------------------------------------------------------------- #


class FakeLLM:
    """Deterministic offline adapter.

    ``plan`` maps ``round -> {expert: decision}`` and lets a test drive the
    consensus loop explicitly. Anything not in the plan is derived from a hash
    of the prompt, so runs stay reproducible without being uniform.
    """

    def __init__(self, plan: dict[int, dict[str, str]] | None = None) -> None:
        self.plan = plan or {}

    async def complete(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        meta: LLMCallMeta | None = None,
    ) -> LLMResult:
        await asyncio.sleep(0)  # keep it a real coroutine

        if purpose == "intent":
            content = json.dumps(self._intent(user), ensure_ascii=False)
        elif purpose == "expert":
            content = json.dumps(self._opinion(meta, user), ensure_ascii=False)
        elif purpose == "meta_decision":
            # 离线**不补差集**：激活集与证据子集完全由规则骨架决定，因此离线 run 的产物与
            # 决策层接入之前逐字一致（差异只留在真实模型链路上）。
            content = json.dumps(
                {
                    "add_experts": [],
                    "weights": {},
                    "evidence_scope": {},
                    "rationale": "离线适配器：不补差集",
                },
                ensure_ascii=False,
            )
        else:
            content = (meta.expert if meta is not None else "") or "system"

        return LLMResult(
            content=content,
            model="fake",
            cached=False,
            usage=Usage(
                calls=1,
                tokens_in=max(1, len(system + user) // 4),
                tokens_out=max(1, len(content) // 4),
            ),
        )

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _evidence_ids(user: str) -> list[str]:
        return sorted(set(_EID_RE.findall(user)))

    def _intent(self, user: str) -> dict[str, Any]:
        """Best-effort request text for the fake's intent branch.

        这条分支在当前管道里没有生产者（``complete_intent`` 不调用 LLM），
        因此请求文本取自 user 消息的第一行——真实意图调用也会把它放在那里。
        """
        request = next((line.strip() for line in user.splitlines() if line.strip()), "")
        return {
            "sub_questions": [
                {"text": f"{request} 的合规性", "discipline": "E03", "priority": 0.7}
            ],
            "query_set": [request] if request else [],
        }

    def _opinion(self, meta: LLMCallMeta | None, user: str) -> dict[str, Any]:
        expert = (meta.expert if meta is not None else "") or "E01"
        round_no = (meta.round if meta is not None else 0) or 1
        evidence_ids = self._evidence_ids(user)

        decision = self.plan.get(round_no, {}).get(expert)
        if decision is None:
            seed = int(hashlib.sha1(f"{expert}|{round_no}|{user[:400]}".encode()).hexdigest(), 16)
            # Later rounds drift toward agreement, mimicking a consensus loop.
            decision = ["approve", "revise", "reject", "approve"][(seed + round_no) % 4]
            if round_no >= 2 and decision == "reject":
                decision = "revise"

        claim_text = f"{expert} 认为该改动在{r'修订后' if round_no >= 2 else r'当前范围内'}可行"
        return {
            "decision": decision,
            "rationale": f"{expert} 基于 {len(evidence_ids)} 条证据给出 {decision}。",
            "evidence_ids": evidence_ids,
            "claims": [
                {
                    "claim": claim_text[:200],
                    "condition": "在既有设计与施工顺序不变的前提下",
                    "evidence_ids": evidence_ids,
                    "discipline": expert,
                }
            ],
            "constraints": [f"{expert}: 需在实施前完成复核"],
            "uncertainties": [f"{expert}: 缺乏历史同类案例" ] if round_no == 1 else [],
            "risk_level": "high" if decision == "reject" else "low",
            "confidence": 0.82,
        }


# --------------------------------------------------------------------------- #
# DeepSeek adapter
# --------------------------------------------------------------------------- #


class DeepSeekLLM:
    """OpenAI-compatible chat completions over httpx."""

    def __init__(
        self,
        cfg: Settings | None = None,
        cache: LLMCache | None = None,
        max_attempts: int = 2,
    ) -> None:
        cfg = cfg or default_settings
        self._api_key = cfg.require_api_key()
        self._base_url = cfg.deepseek_base_url.rstrip("/")
        self._model = cfg.deepseek_model
        self._timeout = cfg.llm_timeout_s
        self._max_attempts = max_attempts
        self._params: dict[str, Any] = {
            "temperature": cfg.llm_temperature,
            "max_tokens": cfg.llm_max_tokens,
            **cfg.thinking_payload,  # {"thinking": {"type": "disabled"}}
        }
        self._cache = cache or LLMCache(cfg.cache_dir, cfg.cache_enabled)
        self._client: httpx.AsyncClient | None = None

    async def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        meta: LLMCallMeta | None = None,
    ) -> LLMResult:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self._params,
        }
        # `meta` is deliberately absent from `body`: it is our bookkeeping, not
        # something the provider should see. It must, however, be part of the
        # cache key — see LLMCache.key.
        cache_key = LLMCache.key(self._model, system, user, self._params, meta)
        hit = self._cache.get(cache_key)
        if hit is not None:
            return hit.model_copy(update={"cached": True})

        last: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                result = await self._post(body)
            except TransientError as exc:
                last = exc
                if attempt >= self._max_attempts:
                    raise
                await asyncio.sleep(exc.retry_after or 0.5 * attempt)
                continue
            self._cache.put(cache_key, result)
            return result
        raise last if last else RuntimeError("unreachable")

    async def _post(self, body: dict[str, Any]) -> LLMResult:
        client = await self._client_or_create()
        try:
            response = await client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise TransientError(f"DeepSeek 超时: {exc}") from exc
        except httpx.TransportError as exc:
            raise TransientError(f"DeepSeek 连接失败: {exc}") from exc

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimited(
                "DeepSeek 限流 429",
                retry_after=float(retry_after) if retry_after else None,
            )
        if response.status_code >= 500:
            raise TransientError(f"DeepSeek 服务端错误 {response.status_code}")
        if response.status_code >= 400:
            raise PermanentExternalError(
                f"DeepSeek 请求被拒 {response.status_code}: {response.text[:300]}"
            )

        try:
            data = response.json()
            message = data["choices"][0]["message"]
        except Exception as exc:
            raise PermanentExternalError(f"DeepSeek 响应结构异常: {exc}") from exc

        usage = data.get("usage") or {}
        return LLMResult(
            content=message.get("content") or "",
            reasoning=message.get("reasoning_content") or "",
            model=data.get("model", self._model),
            cached=False,
            usage=Usage(
                calls=1,
                tokens_in=int(usage.get("prompt_tokens", 0)),
                tokens_out=int(usage.get("completion_tokens", 0)),
            ),
        )