"""LlamaIndex RAG 管道（本项目 ``rag/`` 的真实实现）。

模块分工
--------
| 文件 | 职责 |
| --- | --- |
| ``corpus.py`` | 语料解析：一单一 Document，字段结构化 + ``disciplines`` 打标 |
| ``embeddings.py`` | ``ZhipuEmbedding``（智谱 embedding-3）/ ``FakeEmbedding``（离线） |
| ``llm.py`` | ``LlamaLLMBridge``：把项目的 ``LLMPort`` 接到 LlamaIndex 的 ``CustomLLM`` |
| ``extractors.py`` | ``DomainTripletExtractor``：确定性领域三元组（``kg_nodes``/``kg_relations``） |
| ``graph_store.py`` | Neo4j 领域 schema 的建约束 / 写入 / 清空 |
| ``vector_store.py`` | Qdrant 集合与 payload 索引的显式创建 |
| ``ingest.py`` | ``IngestionPipeline`` 编排 + 命令行（``python -m ec_renew.ingest``） |
| ``retriever.py`` | ``LlamaIndexRetriever``：混合检索，实现 ``ports.RetrieverPort`` |
| ``factory.py`` | 检索后端选择与显式降级报告 |

对外入口只有两个：

* 检索 —— ``from ec_renew.rag_llama import build_retriever``
* 摄取 —— ``from ec_renew.rag_llama.ingest import ingest``

依赖方向：``rag_llama`` 依赖 ``config`` / ``contracts`` / ``errors`` / ``ports``
以及 ``rag.py``（复用其中的 Cypher 与「部门→专业」映射）；``rag.py`` 不 import
本包，因此不存在循环依赖。
"""

from __future__ import annotations

__all__ = ["LlamaIndexRetriever", "build_retriever"]


def __getattr__(name: str):
    """惰性导入：不装 llama-index 时，``--offline`` 路径依然可用。"""
    if name == "build_retriever":
        from .factory import build_retriever

        return build_retriever
    if name == "LlamaIndexRetriever":
        from .retriever import LlamaIndexRetriever

        return LlamaIndexRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
