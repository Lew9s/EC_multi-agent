"""混合检索：Qdrant 语义召回 + Neo4j 结构化召回，RRF 融合。

对应 design.md §5.2.2 的 ``prefetch()``（外环的共享事实基线）：
外环每轮只调一次，结果被冻结成所有专家共同看到的那一份证据。

为什么是混合而不是纯向量
------------------------
* **向量腿**负责「表述不同但意思相近」——用户问「污水井加厚板」，
  历史单里写的是「更换加厚板 结构修改如下」，字面完全对不上；
* **图腿**负责「结构化归属」——``disciplines``（部门→专业）本来就是
  从图里推出来的，纯向量腿拿不到，而它决定后续激活哪些专家。

融合用 **RRF**（``1/(k+rank)``）而不是分数加权：向量余弦与图匹配计数
不在同一个量纲上，归一化会引入一个说不清来历的系数；RRF 只用**排名**，
无需调参，且对两条腿的分数尺度不敏感。

证据来源的诚实性
----------------
只有被图腿匹配到的案例才算 ``source="graph"``（= 真实历史案例，会触发
``assess_grounding`` 的 history 分支）。纯向量命中记为 ``source="text"``：
它确实是检索到的文本，但不能仅凭「最近的 k 条」就宣称有历史依据 ——
向量检索**总是**会返回 k 条，哪怕一条都不相关。
需要让高相似度命中也算历史依据时，把 ``VECTOR_MIN_SCORE`` 设为正数
（例如智谱 embedding-3 可试 0.5）。**默认关闭**，即不夸大保证等级。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from typing import Any

from llama_index.core import VectorStoreIndex

from ..config import Settings
from ..config import settings as default_settings
from ..contracts import (
    EntityRef,
    EvidenceBundle,
    EvidenceMeta,
    GraphExpansion,
)
from ..errors import PermanentExternalError
from ..rag import Neo4jRetriever, case_row_to_meta
from .corpus import ChangeOrder, corpus_path, load_change_orders
from .embeddings import build_embed_model
from .vector_store import build_client, build_vector_store


def rrf_fuse(rankings: Iterable[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion。

    只用排名，因此不需要给两条腿的分数做归一化。``k`` 起平滑作用：
    ``k`` 越大，头部排名的优势越被抹平（经验值 60）。
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return scores


def rank_graph_rows(rows: Sequence[dict[str, Any]], terms: Sequence[str]) -> list[str]:
    """图腿排序：命中查询词的个数降序，同分按 case_id —— 完全确定。"""
    needles = [t.strip() for t in terms if t and t.strip()]
    scored: list[tuple[int, str]] = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            continue
        haystack = " ".join(
            [
                case_id,
                " ".join(row.get("components") or []),
                " ".join(row.get("reasons") or []),
                " ".join(row.get("timepoints") or []),
                " ".join(row.get("departments") or []),
            ]
        )
        matches = sum(1 for needle in needles if needle and needle in haystack)
        scored.append((matches, case_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [case_id for _, case_id in scored]


class LlamaIndexRetriever:
    """``ports.RetrieverPort`` 的真实实现（Qdrant + Neo4j）。"""

    def __init__(
        self,
        cfg: Settings | None = None,
        *,
        offline: bool = False,
        embed_model: Any | None = None,
    ) -> None:
        self._cfg = cfg or default_settings
        self._offline = offline
        self._graph = Neo4jRetriever(self._cfg)
        self._embed = embed_model or build_embed_model(self._cfg, offline=offline)
        self._client = build_client(self._cfg)
        self._index = VectorStoreIndex.from_vector_store(
            build_vector_store(self._client, self._cfg.qdrant_collection),
            embed_model=self._embed,
        )
        self._orders = self._load_orders()

    # -- lifecycle ---------------------------------------------------------- #
    def _load_orders(self) -> dict[str, ChangeOrder]:
        """单号 -> 变更单，用来把证据正文还原成**原文**。

        图谱里只存结构化字段，向量库里的正文是原文；让两条腿取到同一个
        字符串，内容寻址的 EvidenceRegistry 才会把它们归并成同一条证据。
        """
        try:
            path = corpus_path(self._cfg.data_dir, self._cfg.corpus_file)
            orders = load_change_orders(path, separator=self._cfg.corpus_separator)
        except FileNotFoundError:
            return {}
        return {order.case_id: order for order in orders}

    def close(self) -> None:
        self._graph.close()
        self._client.close()
        closer = getattr(self._embed, "close", None)
        if callable(closer):
            closer()

    def health(self) -> dict[str, Any]:
        """可用性探测（factory 用它决定是否降级）。"""
        info: dict[str, Any] = {"backend": "llamaindex", "offline": self._offline}
        self._graph.ping()
        self._client.get_collections()
        count = self._client.count(collection_name=self._cfg.qdrant_collection, exact=True)
        info["points"] = int(count.count)
        info["collection"] = self._cfg.qdrant_collection
        info["min_score"] = self._cfg.vector_min_score
        return info

    # -- port: link / expand (delegated; all Cypher stays in rag.py) -------- #
    async def link(self, request: str) -> list[EntityRef]:
        return await self._graph.link(request)

    async def expand(self, entities: list[EntityRef]) -> GraphExpansion:
        return await self._graph.expand(entities)

    # -- port: prefetch ----------------------------------------------------- #
    async def prefetch(self, queries: list[str], top_k: int) -> EvidenceBundle:
        queries = [q.strip() for q in queries if q and q.strip()]
        if not queries:
            return EvidenceBundle(round=0, warnings=["low_evidence"])

        # 两个阻塞客户端都丢到工作线程，别卡住事件循环（6 个专家靠它并发）。
        vector_task = asyncio.to_thread(self._vector_rankings, queries)
        graph_task = self._graph.cases_by_terms(queries, self._graph_fetch_limit(top_k))
        (vector_rankings, best_score, texts), graph_rows = await asyncio.gather(
            vector_task, graph_task
        )

        graph_rows_by_id = {str(row.get("case_id")): row for row in graph_rows}
        graph_ranking = rank_graph_rows(graph_rows, queries)

        rankings: list[Sequence[str]] = [*vector_rankings, graph_ranking]
        fused = rrf_fuse(rankings, k=self._cfg.rrf_k)

        items: list[EvidenceMeta] = []
        vector_only: list[str] = []
        for case_id, _score in sorted(fused.items(), key=lambda pair: (-pair[1], pair[0])):
            row = graph_rows_by_id.get(case_id)
            score = best_score.get(case_id, 0.0)
            if row is not None:
                meta = case_row_to_meta(row)
                meta = meta.model_copy(update={"content": self._content(case_id, meta.content)})
                meta.score = round(1.0 + score, 4)
            else:
                promoted = self._promote_by_score(score)
                meta = self._text_meta(case_id, texts.get(case_id, ""), score, promote=promoted)
                if not promoted:
                    vector_only.append(case_id)
            items.append(meta)

        items = items[:top_k]
        warnings: list[str] = []
        if not items:
            warnings.append("low_evidence")
        elif vector_only:
            # 显式可见：向量腿找到了一些图腿没匹配上的案例，但它们不计入
            # 历史依据（见模块 docstring）。降级不能是静默的。
            warnings.append(f"vector_only_hits:{len(vector_only)}")
        return EvidenceBundle(
            round=0,
            baseline_ids=[item.evidence_id for item in items],
            items=items,
            warnings=warnings,
        )

    # -- legs --------------------------------------------------------------- #
    def _graph_fetch_limit(self, top_k: int) -> int:
        # 多取一些再在 Python 侧排序：Cypher 的 ORDER BY case_id 只是为了让
        # 截断确定，不携带相关性。留 4 倍余量给排序器。
        return max(self._cfg.graph_top_k, top_k * 4, 50)

    def _vector_rankings(
        self, queries: Sequence[str]
    ) -> tuple[list[list[str]], dict[str, float], dict[str, str]]:
        """每条 query 一个排名列表 + 每个案例的最好分数与正文。"""
        rankings: list[list[str]] = []
        best_score: dict[str, float] = {}
        texts: dict[str, str] = {}
        for query in queries:
            try:
                nodes = self._index.as_retriever(
                    similarity_top_k=self._cfg.top_k
                ).retrieve(query)
            except PermanentExternalError:
                raise
            except Exception as exc:  # qdrant / httpx -> 本项目异常体系
                raise PermanentExternalError(f"向量检索失败: {exc}") from exc

            ranked: list[str] = []
            for item in nodes:
                metadata = getattr(item.node, "metadata", {}) or {}
                case_id = str(metadata.get("case_id") or "").strip()
                if not case_id:
                    continue
                if case_id not in ranked:
                    ranked.append(case_id)
                score = float(item.score or 0.0)
                if score > best_score.get(case_id, float("-inf")):
                    best_score[case_id] = score
                    texts[case_id] = item.node.get_content()
            rankings.append(ranked)
        return rankings, best_score, texts

    # -- evidence construction ---------------------------------------------- #
    def _content(self, case_id: str, fallback: str) -> str:
        """证据正文优先取语料原文（可回溯、且两条腿归并成同一条证据）。"""
        order = self._orders.get(case_id)
        return order.text if order is not None else fallback

    def _text_meta(self, case_id: str, text: str, score: float, *, promote: bool = False) -> EvidenceMeta:
        order = self._orders.get(case_id)
        content = self._content(case_id, text or f"{case_id}（向量命中，图谱中无此单）")
        return EvidenceMeta(
            evidence_id=f"CASE-{case_id}",
            entity_kinds=["CHANGE_ORDER"] if order else [],
            entity_keys=order.entity_keys if order else [case_id],
            disciplines=list(order.disciplines) if order else [],
            group_keys=[order.group_key] if order else [case_id],
            # promote=True 表示相似度已过阈值，这条足以支撑 history_backed。
            source="graph" if promote else "text",
            score=round(score, 4),
            content=content,
        )

    def _promote_by_score(self, score: float) -> bool:
        """分数足够高时，允许纯向量命中算作历史依据（默认关闭）。"""
        return self._cfg.vector_min_score > 0.0 and score >= self._cfg.vector_min_score
