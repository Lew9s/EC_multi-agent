"""向量模型适配层。

``ZhipuEmbedding`` —— 智谱 ``embedding-3``（OpenAI 兼容协议，httpx 直连）。
``FakeEmbedding``  —— 离线确定性向量，无需 key，供测试与 ``--offline`` 使用。

两条硬约束（与 ``llm.py`` 的写法保持一致）：

1. **key 不进业务层**。明文只在本模块内向 ``config`` 取一次，存进
   ``PrivateAttr``（pydantic 的私有属性不进 ``repr`` / ``model_dump``），
   再由 httpx client 的请求头持有。
2. **异常只在边界翻译一次**。上层永远看不到 ``httpx.*``，
   只会看到 ``TransientError`` / ``RateLimited`` / ``PermanentExternalError``。

``FakeEmbedding`` 不是随机数：它是一个字符级 hashing 向量器，因此
「文本越像 → 余弦越接近」，离线跑出来的检索结果是有意义的，而不是噪声。
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections.abc import Sequence
from typing import Any

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import ConfigDict, PrivateAttr

from ..config import Settings
from ..config import settings as default_settings
from ..errors import PermanentExternalError, RateLimited, TransientError

# embedding-3 官方支持的输出维度；换维度等于换索引，必须重建集合。
SUPPORTED_DIMENSIONS = (2048, 1024, 512, 256)


def _lexical_vector(text: str, dim: int) -> list[float]:
    """字符 unigram + bigram 的 hashing 向量，确定性且无第三方依赖。

    中文没有空格分词，用字符 n-gram 反而比词更稳。L2 归一化后
    点积即余弦，正好匹配 Qdrant 的 Cosine 距离。
    """
    vector = [0.0] * dim
    normalized = " ".join(text.split())
    grams: list[str] = []
    for index, char in enumerate(normalized):
        if char.strip():
            grams.append(char)
            if index + 1 < len(normalized) and normalized[index + 1].strip():
                grams.append(normalized[index : index + 2])
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        # 空文本：返回一个确定性的单位向量，避免 NaN。
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class FakeEmbedding(BaseEmbedding):
    """离线向量模型：确定性、无网络、无需 key。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "fake-lexical"
    dimensions: int = 256

    def __init__(self, dimensions: int = 256, **kwargs: Any) -> None:
        super().__init__(dimensions=dimensions, **kwargs)

    def _embed(self, text: str) -> list[float]:
        return _lexical_vector(text, self.dimensions)

    # -- sync -------------------------------------------------------------- #
    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    def _get_text_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    # -- async ------------------------------------------------------------- #
    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def _aget_text_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)


class ZhipuEmbedding(BaseEmbedding):
    """智谱 ``embedding-3``，走 OpenAI 兼容的 ``/embeddings``。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    dimensions: int = 2048
    timeout_s: float = 60.0
    max_attempts: int = 2

    # 密钥与连接对象都放私有属性：不进 repr、不进 model_dump、不进事件日志。
    _api_key: str = PrivateAttr(default="")
    _client: httpx.Client | None = PrivateAttr(default=None)
    _aclient: httpx.AsyncClient | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        cfg: Settings | None = None,
        dimensions: int | None = None,
        embed_batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        cfg = cfg or default_settings
        dimensions = dimensions or cfg.embedding_dimensions
        if dimensions not in SUPPORTED_DIMENSIONS:
            raise PermanentExternalError(
                f"embedding-3 不支持 {dimensions} 维；可选 {SUPPORTED_DIMENSIONS}。"
                "改维度后必须重建 Qdrant 集合。"
            )
        super().__init__(
            model_name=cfg.embedding_model,
            embed_batch_size=embed_batch_size or cfg.embedding_batch_size,
            api_base=cfg.zhipu_base_url.rstrip("/"),
            dimensions=dimensions,
            timeout_s=cfg.embedding_timeout_s,
            **kwargs,
        )
        # 唯一一次取明文；业务层永远拿不到。
        self._api_key = cfg.require_zhipu_api_key()

    # -- transport --------------------------------------------------------- #
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _sync_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.api_base, timeout=self.timeout_s, headers=self._headers()
            )
        return self._client

    def _async_client(self) -> httpx.AsyncClient:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(
                base_url=self.api_base, timeout=self.timeout_s, headers=self._headers()
            )
        return self._aclient

    def close(self) -> None:
        """关掉同步连接。异步连接必须走 ``aclose``（不能在无事件循环时关）。"""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None
        self.close()

    # -- exception translation (the only place httpx.* is visible) --------- #
    @staticmethod
    def _translate(response: httpx.Response) -> None:
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimited(
                "智谱 embedding 限流 429",
                retry_after=float(retry_after) if retry_after else None,
            )
        if response.status_code >= 500:
            raise TransientError(f"智谱 embedding 服务端错误 {response.status_code}")
        if response.status_code >= 400:
            raise PermanentExternalError(
                f"智谱 embedding 请求被拒 {response.status_code}: {response.text[:300]}"
            )

    @staticmethod
    def _parse(payload: dict[str, Any]) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise PermanentExternalError(f"智谱 embedding 响应结构异常: {str(payload)[:300]}")
        # 按 index 排序：批量返回不保证顺序，乱序会让向量和文本错配。
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise PermanentExternalError("智谱 embedding 返回了空向量")
            vectors.append([float(value) for value in vector])
        return vectors

    def _post(self, client: httpx.Client, texts: list[str]) -> list[list[float]]:
        body = {"model": self.model_name, "input": texts, "dimensions": self.dimensions}
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = client.post("/embeddings", json=body)
            except httpx.TimeoutException as exc:
                last = TransientError(f"智谱 embedding 超时: {exc}")
            except httpx.TransportError as exc:
                last = TransientError(f"智谱 embedding 连接失败: {exc}")
            else:
                try:
                    self._translate(response)
                except TransientError as exc:
                    # 429 / 5xx 属于可重试：必须落回循环，而不是直接抛出去。
                    last = exc
                else:
                    try:
                        return self._parse(response.json())
                    except ValueError as exc:
                        raise PermanentExternalError(f"智谱 embedding 响应非 JSON: {exc}") from exc
            if attempt < self.max_attempts:
                delay = getattr(last, "retry_after", None) or 0.5 * attempt
                time.sleep(max(0.0, delay))
        raise last if last else RuntimeError("unreachable")

    async def _apost(self, texts: list[str]) -> list[list[float]]:
        client = self._async_client()
        body = {"model": self.model_name, "input": texts, "dimensions": self.dimensions}
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await client.post("/embeddings", json=body)
            except httpx.TimeoutException as exc:
                last = TransientError(f"智谱 embedding 超时: {exc}")
            except httpx.TransportError as exc:
                last = TransientError(f"智谱 embedding 连接失败: {exc}")
            else:
                try:
                    self._translate(response)
                except TransientError as exc:
                    # 429 / 5xx 属于可重试：必须落回循环，而不是直接抛出去。
                    last = exc
                else:
                    try:
                        return self._parse(response.json())
                    except ValueError as exc:
                        raise PermanentExternalError(f"智谱 embedding 响应非 JSON: {exc}") from exc
            if attempt < self.max_attempts:
                await asyncio.sleep(getattr(last, "retry_after", None) or 0.5 * attempt)
        raise last if last else RuntimeError("unreachable")

    # -- BaseEmbedding ------------------------------------------------------ #
    def _get_text_embedding(self, text: str) -> list[float]:
        return self._post(self._sync_client(), [text])[0]

    def _get_text_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        return self._post(self._sync_client(), list(texts))

    def _get_query_embedding(self, query: str) -> list[float]:
        # embedding-3 对 query/document 不区分任务类型，走同一条路径。
        return self._post(self._sync_client(), [query])[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return (await self._apost([text]))[0]

    async def _aget_text_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._apost(list(texts))

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return (await self._apost([query]))[0]


def build_embed_model(cfg: Settings | None = None, *, offline: bool = False) -> BaseEmbedding:
    """选向量模型。

    没有 key **不等于**可以静默降级（design.md P5）。这里只有调用方显式要求
    ``offline=True`` 才回退到 ``FakeEmbedding``，否则直接报错，让配置问题
    在启动时就暴露，而不是变成一堆看起来正常的空检索。
    """
    cfg = cfg or default_settings
    if offline:
        # 维度必须与集合一致，否则查询维度对不上。
        return FakeEmbedding(dimensions=cfg.embedding_dimensions)
    return ZhipuEmbedding(cfg=cfg)
