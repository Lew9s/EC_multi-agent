"""RAG 管道：领域图 + LlamaIndex 摄取与混合检索（design.md §3 模块 5）。

模块分工
--------
| 文件 | 职责 |
| --- | --- |
| ``graph.py`` | 领域图与 Cypher（``Neo4jRetriever`` / ``InMemoryRetriever``）+「部门→专业」映射 |
| ``corpus.py`` | 语料解析：一单一 Document，字段结构化 + ``disciplines`` 打标 |
| ``embeddings.py`` | ``ZhipuEmbedding``（智谱 embedding-3）/ ``FakeEmbedding``（离线） |
| ``llm_bridge.py`` | ``LlamaLLMBridge``：把项目的 ``LLMPort`` 接到 LlamaIndex 的 ``CustomLLM`` |
| ``extractors.py`` | ``DomainTripletExtractor``：确定性领域三元组（``kg_nodes``/``kg_relations``） |
| ``graph_store.py`` | Neo4j 领域 schema 的建约束 / 写入 / 清空 |
| ``vector_store.py`` | Qdrant 集合与 payload 索引的显式创建 |
| ``ingest.py`` | ``IngestionPipeline`` 编排 + 命令行（``python -m ec_renew.rag.ingest``） |
| ``retriever.py`` | ``LlamaIndexRetriever``：混合检索（RRF），实现 ``ports.RetrieverPort`` |
| ``factory.py`` | 检索后端选择与显式降级报告 |
| ``calibrate.py`` | 向量阈值标定（``python -m ec_renew.rag.calibrate``） |

对外入口
--------
* 领域图与映射 —— ``from ec_renew.rag import InMemoryRetriever, disciplines_for_departments``
* 检索 —— ``from ec_renew.rag import build_retriever``
* 摄取 —— ``from ec_renew.rag.ingest import ingest``

后两个走惰性导入：不装 llama-index 时，``--offline`` 路径依然可用。

依赖方向：``rag`` 依赖 ``config`` / ``contracts`` / ``errors`` / ``ports`` / ``llm``，
**不 import** ``workflow`` / ``agents`` / ``interface``（单向依赖，见 AGENTS.md §1.2）。
"""

from __future__ import annotations

from .graph import (
    DEFAULT_DISCIPLINE,
    DEPT_TO_DISCIPLINE,
    InMemoryRetriever,
    Neo4jRetriever,
    case_row_to_meta,
    disciplines_for_departments,
    is_partial_identifier,
)

__all__ = [
    "DEFAULT_DISCIPLINE",
    "DEPT_TO_DISCIPLINE",
    "InMemoryRetriever",
    "LlamaIndexRetriever",
    "Neo4jRetriever",
    "build_retriever",
    "case_row_to_meta",
    "disciplines_for_departments",
    "is_partial_identifier",
]


def __getattr__(name: str):
    """惰性导入：不装 llama-index 时，``--offline`` 路径依然可用。"""
    if name == "build_retriever":
        from .factory import build_retriever

        return build_retriever
    if name == "LlamaIndexRetriever":
        from .retriever import LlamaIndexRetriever

        return LlamaIndexRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
