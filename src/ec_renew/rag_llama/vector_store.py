"""Qdrant 向量库接入（docker compose 部署）。

集合由本项目**显式创建**，不让 ``QdrantVectorStore`` 在首次写入时隐式建：
向量维度一旦和 ``EMBEDDING_DIMENSIONS`` 不一致，检索会静默地什么都查不到，
所以宁可在启动阶段就报错（design.md P5：降级必须显式）。

``disciplines`` / ``case_id`` / ``group_key`` 建 payload 索引 —— 元数据过滤
必须有索引，否则 Qdrant 会退化成全表扫描。
"""

from __future__ import annotations

from typing import Any

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..config import Settings
from ..config import settings as default_settings
from ..errors import PermanentExternalError, TransientError

# 需要建索引的 payload 字段：检索时的过滤条件都落在这几个键上。
PAYLOAD_INDEX_FIELDS: tuple[str, ...] = ("case_id", "group_key", "disciplines", "component")


def _translate(exc: Exception) -> Exception:
    name = type(exc).__name__
    if "ResponseHandlingException" in name or "ConnectError" in name or "Timeout" in name:
        return TransientError(f"Qdrant 不可用: {exc}")
    return PermanentExternalError(f"Qdrant 操作失败: {exc}")


def build_client(cfg: Settings | None = None) -> QdrantClient:
    cfg = cfg or default_settings
    kwargs: dict[str, Any] = {"url": cfg.qdrant_url, "timeout": 30.0}
    api_key = cfg.qdrant_api_key.get_secret_value()
    if api_key:
        kwargs["api_key"] = api_key
    try:
        return QdrantClient(**kwargs)
    except Exception as exc:
        raise _translate(exc) from exc


def collection_dimensions(client: QdrantClient, name: str) -> int | None:
    """已存在集合的向量维度；集合不存在返回 ``None``。"""
    try:
        if not client.collection_exists(name):
            return None
        info = client.get_collection(name)
    except Exception as exc:
        raise _translate(exc) from exc
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):  # 命名词向量：本项目只用匿名单向量
        if not vectors:
            return None
        return int(next(iter(vectors.values())).size)
    return int(vectors.size)


def ensure_collection(
    client: QdrantClient,
    name: str,
    dimensions: int,
    *,
    recreate: bool = False,
) -> dict[str, Any]:
    """建集合 + payload 索引；已存在则校验维度。"""
    try:
        existing = collection_dimensions(client, name)
        if existing is not None and recreate:
            client.delete_collection(name)
            existing = None
        if existing is None:
            client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=dimensions, distance=qmodels.Distance.COSINE
                ),
            )
            created = True
        elif existing != dimensions:
            raise PermanentExternalError(
                f"集合 {name!r} 的向量维度是 {existing}，"
                f"而 EMBEDDING_DIMENSIONS={dimensions}。"
                "两者必须一致：改维度请重建集合（ingest --recreate），"
                "或把 EMBEDDING_DIMENSIONS 改回去。"
            )
        else:
            created = False

        for field in PAYLOAD_INDEX_FIELDS:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        count = int(client.count(collection_name=name, exact=True).count)
    except PermanentExternalError:
        raise
    except Exception as exc:
        raise _translate(exc) from exc
    return {"collection": name, "created": created, "dimensions": dimensions, "points": count}


def build_vector_store(
    client: QdrantClient, name: str, *, text_key: str = "text"
) -> QdrantVectorStore:
    """把同一个 client 交给 LlamaIndex，避免两条连接看到不同的状态。"""
    return QdrantVectorStore(
        client=client,
        collection_name=name,
        text_key=text_key,
        # 集合由 ensure_collection 建好，这里不再重复建（避免隐式维度推断）。
        dense_config=None,
    )


def collection_stats(client: QdrantClient, name: str) -> dict[str, Any]:
    try:
        if not client.collection_exists(name):
            return {"collection": name, "exists": False, "points": 0, "dimensions": None}
        info = client.get_collection(name)
        return {
            "collection": name,
            "exists": True,
            "points": int(client.count(collection_name=name, exact=True).count),
            "dimensions": collection_dimensions(client, name),
            "status": str(getattr(info, "status", "")),
        }
    except Exception as exc:
        raise _translate(exc) from exc


def ping(client: QdrantClient) -> bool:
    client.get_collections()
    return True
