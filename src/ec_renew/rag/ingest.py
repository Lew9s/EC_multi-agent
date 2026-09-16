"""摄取管道：语料 -> Qdrant（向量）+ Neo4j（图）。

一条 ``IngestionPipeline`` 同时干两件事：

    documents ──[DomainTripletExtractor]──> kg_nodes / kg_relations ──> Neo4j 领域图
              └─[ZhipuEmbedding]─────────> 向量 ──────────────────────> Qdrant

两条支路共用同一批 node（``Document.id_`` 决定 node id），所以：
* 重复摄取是**覆盖**而不是追加（幂等）；
* 向量命中携带 ``case_id``，可以回查图谱拿到 departments/disciplines。

整个摄取是阻塞的（httpx + neo4j driver 都是同步），对外只暴露
``ingest`` 一个 async 壳，内部统一丢到工作线程，避免阻塞事件循环。

命令行：
    python -m ec_renew.rag.ingest --offline          # 无 key 自检（写入假向量）
    python -m ec_renew.rag.ingest                    # 真实摄取（需要两个 key）
    python -m ec_renew.rag.ingest --recreate --wipe  # 重建集合与图
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..config import settings as default_settings
from ..observability import JsonlEventLog, NullEventLog
from .corpus import ChangeOrder, corpus_path, load_change_orders, summarize, to_documents
from .embeddings import build_embed_model
from .extractors import build_kg_extractor
from .graph_store import Neo4jDomainWriter, WriteReport, triples_from_nodes
from .vector_store import (
    build_client,
    build_vector_store,
    collection_stats,
    ensure_collection,
)


@dataclass
class IngestReport:
    """一次摄取的完整结果。确定性 —— 可用来比对两次摄取是否一致。"""

    corpus: dict[str, Any] = field(default_factory=dict)
    documents: int = 0
    nodes: int = 0
    vector: dict[str, Any] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)
    extractor: str = "rule"
    embedding_model: str = ""
    embedding_dimensions: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "documents": self.documents,
            "nodes": self.nodes,
            "vector": self.vector,
            "graph": self.graph,
            "extractor": self.extractor,
            "embedding": {
                "model": self.embedding_model,
                "dimensions": self.embedding_dimensions,
            },
        }


def _load_documents(cfg: Settings) -> tuple[list[ChangeOrder], list[Any], dict[str, Any]]:
    path = corpus_path(cfg.data_dir, cfg.corpus_file)
    orders = load_change_orders(path, separator=cfg.corpus_separator)
    return orders, to_documents(orders), summarize(orders)


def ingest_sync(
    cfg: Settings | None = None,
    *,
    offline: bool = False,
    extractor: str = "rule",
    recreate: bool = False,
    wipe: bool = False,
    reset: bool = False,
    events: Any | None = None,
    vector_only: bool = False,
    graph_only: bool = False,
) -> IngestReport:
    """阻塞版摄取。``asyncio`` 调用方请用 :func:`ingest`。"""
    cfg = cfg or default_settings
    sink = events or NullEventLog()
    if vector_only and graph_only:
        raise ValueError("vector_only 与 graph_only 不能同时为真")

    _, documents, corpus_stats = _load_documents(cfg)
    report = IngestReport(
        corpus=corpus_stats,
        documents=len(documents),
        extractor=extractor,
        embedding_model=cfg.embedding_model if not offline else "fake-lexical",
        embedding_dimensions=cfg.embedding_dimensions,
    )
    sink.emit("ingest_started", **report.as_dict())

    embed_model = build_embed_model(cfg, offline=offline)
    kg_extractor = build_kg_extractor(mode=extractor, offline=offline)

    # 按需拼装变换链：--graph-only 就不该付 embedding 的钱，
    # --vector-only 也不该为没有被写入的图跑一遍抽取。
    from llama_index.core.ingestion import IngestionPipeline

    transformations: list[Any] = []
    if not vector_only:
        transformations.append(kg_extractor)
    if not graph_only:
        transformations.append(embed_model)

    client = None
    vector_store = None
    if not graph_only:
        client = build_client(cfg)
        report.vector = ensure_collection(
            client,
            cfg.qdrant_collection,
            cfg.embedding_dimensions,
            recreate=recreate,
        )
        vector_store = build_vector_store(client, cfg.qdrant_collection)

    pipeline = IngestionPipeline(transformations=transformations, vector_store=vector_store)
    nodes = pipeline.run(documents=documents, show_progress=False)
    report.nodes = len(nodes)
    if client is not None:
        # ensure_collection 里的计数发生在写入之前；这里补一次真实点数。
        report.vector.update(collection_stats(client, cfg.qdrant_collection))

    # ---- 图支路 ----------------------------------------------------------- #
    if not vector_only:
        writer = Neo4jDomainWriter(cfg)
        try:
            writer.ping()
            if wipe:
                writer.wipe_all()
                sink.emit("graph_wiped", scope="all")
            elif reset:
                writer.reset_domain()
                sink.emit("graph_reset", scope="domain")
            writer.ensure_schema()
            triples = triples_from_nodes(list(nodes))
            write_report: WriteReport = writer.write_triples(triples)
            report.graph = {
                "triples": len(triples),
                **write_report.as_dict(),
                "counts": writer.counts(),
            }
        finally:
            writer.close()

    sink.emit("ingest_finished", **report.as_dict())
    return report


async def ingest(cfg: Settings | None = None, **kwargs: Any) -> IngestReport:
    """异步壳：所有阻塞 IO 都在工作线程里。"""
    return await asyncio.to_thread(ingest_sync, cfg, **kwargs)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ec_renew.rag.ingest",
        description="把变更单语料摄取进 Qdrant（向量）与 Neo4j（领域图）",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="用 FakeEmbedding（无需 ZHIPU_API_KEY）。注意：会写入假向量，仅用于自检",
    )
    parser.add_argument(
        "--extractor",
        default=None,
        choices=["rule", "llm"],
        help="三元组抽取器（默认取 .env 的 KG_EXTRACTOR，通常为 rule）",
    )
    parser.add_argument("--recreate", action="store_true", help="删除并重建 Qdrant 集合")
    parser.add_argument("--reset", action="store_true", help="清空本项目的领域图后再写")
    parser.add_argument("--wipe", action="store_true", help="清空整个 Neo4j 库后再写（迁移用）")
    parser.add_argument("--vector-only", action="store_true", help="只写向量，不动图")
    parser.add_argument("--graph-only", action="store_true", help="只写图，不动向量")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = default_settings
    if not args.offline and not cfg.zhipu_api_key.get_secret_value():
        print(
            "ZHIPU_API_KEY 未设置。请在 .env 里填写，或先用 --offline 自检。",
            file=sys.stderr,
        )
        return 2

    events = JsonlEventLog(cfg.events_path, "ingest")
    try:
        report = ingest_sync(
            cfg,
            offline=args.offline,
            extractor=args.extractor or cfg.kg_extractor,
            recreate=args.recreate,
            wipe=args.wipe,
            reset=args.reset,
            vector_only=args.vector_only,
            graph_only=args.graph_only,
            events=events,
        )
    except FileNotFoundError as exc:
        # 语料不属于版本库（见 data/README.md），第一次跑很容易撞上。
        # 给一句能照做的提示，而不是丢一个 traceback。
        print(f"{exc}\n\n语料不随仓库分发。请按 data/README.md 的格式自备，"
              f"或把 DATA_DIR / CORPUS_FILE 指向你的文件。", file=sys.stderr)
        return 2
    finally:
        events.close()

    payload = report.as_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("摄取完成")
        print(f"  语料：{payload['corpus']}")
        print(f"  文档/节点：{payload['documents']} / {payload['nodes']}")
        print(f"  向量：{payload['vector']}")
        print(f"  图：{payload['graph']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
