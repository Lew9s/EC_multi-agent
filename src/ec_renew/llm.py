"""LLM adapters.

* ``FakeLLM``      — offline, deterministic, no API key. Used to prove the
                     pipeline end-to-end before spending money.
* ``DeepSeekLLM``  — OpenAI-compatible endpoint over ``httpx`` (the ``openai``
                     SDK is not installed on this lab machine). The extra body
                     field ``{"thinking": {"type": ...}}`` is passed verbatim.

Both go through ``LLMCache``: content-addressed, so re-running an experiment is
free and byte-reproducible.
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
from .contracts import LLMResult, Usage
from .errors import PermanentExternalError, RateLimited, TransientError

# --------------------------------------------------------------------------- #
# Prompt context header
# --------------------------------------------------------------------------- #

_CTX_RE = re.compile(r"\[\[CTX\s+(?P<body>[^\]]*)\]\]")
_EID_RE = re.compile(r"E-[0-9a-f]{12}")


def ctx_header(**kv: object) -> str:
    """Machine-readable context line embedded in prompts.

    Keeps the fake adapter honest (it reads the same context the real model
    does) and makes event logs self-describing.
    """
    def _clean(value: object) -> str:
        return str(value).replace(" ", "_").replace("\n", "_")

    body = " ".join(f"{k}={_clean(v)}" for k, v in kv.items())
    return f"[[CTX {body}]]"


def parse_ctx(text: str) -> dict[str, str]:
    match = _CTX_RE.search(text)
    if not match:
        return {}
    out: dict[str, str] = {}
    for part in match.group("body").split():
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


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
    def key(model: str, system: str, user: str, params: dict[str, Any]) -> str:
        payload = json.dumps(
            {"model": model, "system": system, "user": user, "params": params},
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

    async def complete(self, *, purpose: str, system: str, user: str) -> LLMResult:
        await asyncio.sleep(0)  # keep it a real coroutine
        ctx = parse_ctx(user) or parse_ctx(system)

        if purpose == "intent":
            content = json.dumps(self._intent(ctx, user), ensure_ascii=False)
        elif purpose == "expert":
            content = json.dumps(self._opinion(ctx, user), ensure_ascii=False)
        else:
            content = str(ctx.get("expert", "system"))

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

    def _intent(self, ctx: dict[str, str], user: str) -> dict[str, Any]:
        request = ctx.get("request", "")
        return {
            "sub_questions": [
                {"text": f"{request} 的合规性", "discipline": "E03", "priority": 0.7}
            ],
            "query_set": [request] if request else [],
        }

    def _opinion(self, ctx: dict[str, str], user: str) -> dict[str, Any]:
        expert = ctx.get("expert", "E01")
        round_no = int(ctx.get("round", "1") or 1)
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

    async def complete(self, *, purpose: str, system: str, user: str) -> LLMResult:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self._params,
        }
        cache_key = LLMCache.key(self._model, system, user, self._params)
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