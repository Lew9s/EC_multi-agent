"""检索后端选择 + **显式**降级。

design.md P5「降级必须显式」：任何一次回退都要带上原因，交给调用方打印并写进
事件日志。这里绝不「悄悄换成内存检索却假装一切正常」—— 那会让一次
「检索不到」看起来像「本来就没有历史案例」。

四档：

| mode | 组成 | 外部依赖 |
| --- | --- | --- |
| ``llamaindex`` | Qdrant 向量 + Neo4j 图（RRF 融合） | 两个都要 |
| ``graph`` | 仅 Neo4j 结构化检索 | Neo4j |
| ``memory`` | 确定性内存 fixture | 无 |
| ``auto`` | 依次探测，取第一个可用的，并把每次降级记进 notes | — |
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..config import Settings
from ..config import settings as default_settings
from ..errors import InvalidRequest, InvariantViolation
from ..ports import EventSinkPort, RetrieverPort
from ..rag import InMemoryRetriever, Neo4jRetriever

VALID_MODES: tuple[str, ...] = ("auto", "llamaindex", "graph", "memory")


def _try_llamaindex(cfg: Settings, offline: bool, notes: list[str]) -> Any | None:
    from .retriever import LlamaIndexRetriever

    retriever: Any | None = None
    try:
        retriever = LlamaIndexRetriever(cfg, offline=offline)
        info = retriever.health()
    except InvariantViolation:
        raise
    except Exception as exc:  # noqa: BLE001 — 降级探测必须接住一切外部故障
        if retriever is not None:
            retriever.close()
        notes.append(
            f"降级：LlamaIndex 混合检索不可用（{type(exc).__name__}: {str(exc)[:160]}）"
        )
        return None
    notes.append(
        f"backend=llamaindex｜Qdrant 集合 {info['collection']} 共 {info['points']} 点"
        f"｜VECTOR_MIN_SCORE={info['min_score']}"
    )
    return retriever


def _try_graph(cfg: Settings, notes: list[str]) -> Any | None:
    retriever = Neo4jRetriever(cfg)
    try:
        retriever.ping()
    except InvariantViolation:
        raise
    except Exception as exc:  # noqa: BLE001 — 降级探测必须接住一切外部故障
        retriever.close()
        notes.append(f"降级：Neo4j 不可用（{type(exc).__name__}: {str(exc)[:160]}）")
        return None
    notes.append("backend=graph｜仅 Neo4j 结构化检索（无向量召回）")
    return retriever


def build_retriever(
    cfg: Settings | None = None,
    *,
    mode: str | None = None,
    offline: bool = False,
    events: EventSinkPort | None = None,
) -> tuple[RetrieverPort, list[str]]:
    """构造检索器 + 返回降级说明。

    返回 ``(retriever, notes)``：``notes`` 为空表示没有任何降级。
    调用方负责把 ``notes`` 打印出来并写入事件日志。
    """
    cfg = cfg or default_settings
    mode = (mode or cfg.rag_backend or "auto").strip().lower()
    if mode not in VALID_MODES:
        raise InvalidRequest(f"未知的 RAG_BACKEND: {mode!r}（可选 {' | '.join(VALID_MODES)}）")

    notes: list[str] = []
    retriever: Any

    if mode == "memory":
        retriever = InMemoryRetriever()
        notes.append("backend=memory｜显式指定，不访问任何外部服务")
    elif mode == "graph":
        retriever = _try_graph(cfg, notes)
        if retriever is None:
            # 显式指定就不许偷偷降级：让调用方立刻看到问题。
            raise InvalidRequest("要求 RAG_BACKEND=graph，但 Neo4j 不可用：" + notes[-1])
    elif mode == "llamaindex":
        retriever = _try_llamaindex(cfg, offline, notes)
        if retriever is None:
            raise InvalidRequest("要求 RAG_BACKEND=llamaindex，但依赖不可用：" + notes[-1])
    else:  # auto
        retriever = _try_llamaindex(cfg, offline, notes)
        if retriever is None:
            retriever = _try_graph(cfg, notes)
        if retriever is None:
            retriever = InMemoryRetriever()
            notes.append("最终降级：memory（不访问任何外部服务，结论仅作流程演示）")

    if events is not None:
        events.emit(
            "retriever_selected",
            requested=mode,
            resolved=notes[-1] if notes else mode,
            degraded=len(notes) > 1,
            notes=list(notes),
        )
    return retriever, notes


def describe(notes: Sequence[str]) -> str:
    return "\n".join(f"  · {note}" for note in notes)
