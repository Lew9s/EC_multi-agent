"""RAG 管道（LlamaIndex）的约束测试。

分两类：

* **纯单元**（不需要任何服务）—— 语料解析、向量器、抽取器、融合排序、
  密钥不外泄。这些是回归主力，必须永远能跑。
* **集成**（需要 Qdrant + Neo4j）—— 真摄取一遍再检索一遍。服务不通时
  自动 skip，而不是失败：CI 里没起容器不应该表现为代码有 bug。

集成测试用一个**独立集合名** ``ec_renew_pytest``，跑完删掉，不碰
``QDRANT_COLLECTION`` 指向的真实集合。图侧无法隔离（社区版只有一个
database），但摄取是 MERGE 幂等写，重复跑不会改变图的状态。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import httpx
import pytest
from llama_index.core.schema import TextNode
from pydantic import SecretStr

from ec_renew.config import settings as base_settings
from ec_renew.errors import PermanentExternalError, RateLimited, TransientError
from ec_renew.rag import disciplines_for_departments, is_partial_identifier
from ec_renew.rag_llama.corpus import (
    ChangeOrder,
    document_id,
    load_change_orders,
    parse_change_orders,
    summarize,
    to_documents,
)
from ec_renew.rag_llama.embeddings import (
    FakeEmbedding,
    ZhipuEmbedding,
    build_embed_model,
)
from ec_renew.rag_llama.extractors import (
    DomainTripletExtractor,
    structural_refs,
)
from ec_renew.rag_llama.graph_store import (
    Triple,
    triples_from_nodes,
    validate_triple,
)
from ec_renew.rag_llama.retriever import rank_graph_rows, rrf_fuse

CORPUS = "data/zahuo.txt"
SEPARATOR = r"!@#\$%\^&\*"
SCRATCH_COLLECTION = "ec_renew_pytest"
SAMPLE_COLLECTION = "ec_renew_pytest_sample"
# 合成语料的单号前缀：既与真实语料区分，也是图侧回收的依据
SYNTHETIC_PREFIX = "S-"

REAL_CORPUS = Path(CORPUS)
SAMPLE_CORPUS = Path("tests/fixtures/sample_change_orders.txt")
has_real_corpus = REAL_CORPUS.is_file()
requires_corpus = pytest.mark.skipif(
    not has_real_corpus,
    reason=f"缺少真实语料 {REAL_CORPUS}（业务数据不进版本库，见 data/README.md）",
)


def _orders() -> list[ChangeOrder]:
    return load_change_orders(CORPUS, separator=SEPARATOR)


# --------------------------------------------------------------------------- #
# 语料解析
#
# 真实语料是业务数据，刻意不进版本库（见 data/README.md）。因此这里分两层：
#   * `requires_corpus` —— 断言真实语料**统计量**的用例，缺语料时 skip；
#   * 合成语料 `tests/fixtures/sample_change_orders.txt` —— 断言**行为**的用例，
#     永远运行，CI 靠它覆盖「摄取 → 建图 → 检索」整条链路。
# --------------------------------------------------------------------------- #


@requires_corpus
def test_corpus_parses_every_change_order() -> None:
    orders = _orders()
    assert len(orders) == 118
    assert orders[0].case_id == "H-01"
    stats = summarize(orders)
    assert stats["orders"] == 118
    # 48 = 32 个多成员变更组（E-10-1..E-10-5）+ 16 个本身就是主单号的单例
    # （H-01、H-06…）。单例必须留在自己的组里，不能被错误地并进 "H"。
    assert stats["groups"] == 48
    # 与旧实现（另一套图谱）的计数一致：说明解析没有丢字段。
    assert stats["components"] == 89


@requires_corpus
def test_corpus_keeps_verbatim_text() -> None:
    """证据正文必须是原文，任何解析都不得改写它（可回溯约束的前提）。"""
    order = _orders()[0]
    assert order.text.startswith("单号:H-01")
    # 解析结果与原文并存，而不是二选一
    assert "变更原因" in order.text
    assert order.reason and order.reason in order.text


def test_sample_corpus_parses_deterministically() -> None:
    """合成语料：格式与真实语料一致，用来在无语料环境下守住解析行为。

    单号刻意用 ``S-`` 前缀而**不复用真实语料的 H-\\* 编号**：Neo4j 社区版只有
    一个 database，若编号相同，合成数据会 MERGE 进真实案例，造成不可见的污染。
    """
    orders = load_change_orders(SAMPLE_CORPUS, separator=SEPARATOR)
    assert len(orders) == 8
    by_id = {order.case_id: order for order in orders}

    # 子单归到主单号的组，主单号本身不被误切
    assert by_id["S-02-1"].group_key == "S-02"
    assert by_id["S-02-2"].group_key == "S-02"
    assert by_id["S-05"].group_key == "S-05"
    assert by_id["S-01"].group_key == "S-01"

    # 部门 -> 专业：物供部 -> E06（材料与焊接）；轮机车间 -> E05
    assert "E06" in by_id["S-06-1"].disciplines
    assert "E05" in by_id["S-04-1"].disciplines

    # 缺字段要如实记录，不许编造
    assert "变更时间点" in by_id["S-06-1"].missing
    assert by_id["S-01"].missing == ()

    # 两条腿共用的稳定 id
    assert by_id["S-01"].entity_keys == ["S-01", "缆绳"]
    # 6 个变更组：S-02 / S-03 各含 2 张子单，其余各自成组
    assert summarize(orders)["groups"] == 6


def test_missing_fields_are_recorded_not_invented() -> None:
    orders = parse_change_orders("变更内容:只有一个字段\n!@#$%^&*\n单号:X-9\n变更对象:甲板板", source_file="t.txt")
    assert [o.case_id for o in orders] == ["AUTO-001", "X-9"]
    assert "单号" in orders[0].missing
    assert orders[0].group_key == "AUTO-001"


def test_group_key_strips_only_the_trailing_index() -> None:
    orders = parse_change_orders("单号:H-03-7", source_file="t.txt")
    assert orders[0].group_key == "H-03"
    single = parse_change_orders("单号:H-19", source_file="t.txt")
    assert single[0].group_key == "H-19"


def test_disciplines_are_tagged_at_parse_time() -> None:
    """Q-01(b)：建库时打标，检索期零成本。"""
    orders = parse_change_orders("单号:H-9\n签收部门:物供部，船体车间", source_file="t.txt")
    assert orders[0].disciplines == ("E01", "E06")
    # 未知部门退回质量（E03），保证「任何变更都涉及规范」
    assert disciplines_for_departments(["某个不认识的部门"]) == ["E03"]


def test_document_id_is_a_stable_uuid() -> None:
    first = document_id("zahuo.txt", "H-01")
    assert first == document_id("zahuo.txt", "H-01"), "重复摄取必须落在同一个点上"
    assert first != document_id("zahuo.txt", "H-02")
    # Qdrant 的 point id 只接受 uint64 或 UUID
    assert str(uuid.UUID(first)) == first


@requires_corpus
def test_corpus_metadata_carries_case_id() -> None:
    documents = to_documents(_orders()[:3])
    assert len(documents) == 3
    metadata = documents[1].metadata
    assert metadata["case_id"] == "H-02-1"
    assert metadata["group_key"] == "H-02"
    assert metadata["disciplines"]
    assert metadata["entity_keys"][0] == "H-02-1"


# --------------------------------------------------------------------------- #
# getContent 前缀精度（link 的经典误召回）
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("FR3", "301分段FR36污水井更换加厚板", True),
        ("FR36", "301分段FR36污水井更换加厚板", False),
        ("FR3", "FR3 与 FR36 都在", False),
        ("301分段", "301分段FR36污水井", False),
        ("污水井", "污水井更换", False),
    ],
)
def test_partial_identifier_guard(name: str, text: str, expected: bool) -> None:
    assert is_partial_identifier(name, text) is expected


# --------------------------------------------------------------------------- #
# 向量模型
# --------------------------------------------------------------------------- #


def test_fake_embedding_is_deterministic_and_lexical() -> None:
    model = FakeEmbedding(dimensions=256)
    text = "301分段FR36污水井更换加厚板"
    vector = model.get_text_embedding(text)
    assert len(vector) == 256
    assert vector == model.get_text_embedding(text)
    assert model.get_text_embedding_batch(["a", "b"]) is not None

    def cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    near = model.get_text_embedding("301分段FR36污水井更换加厚板 焊接")
    far = model.get_text_embedding("完全不相干的内容")
    assert cosine(vector, near) > cosine(vector, far)


def test_build_embed_model_refuses_silent_fallback() -> None:
    """没有 key 时必须报错，不能悄悄换成假向量（design.md P5）。"""
    cfg = base_settings.model_copy(update={"zhipu_api_key": SecretStr("")})
    with pytest.raises(RuntimeError):
        build_embed_model(cfg)
    assert isinstance(build_embed_model(cfg, offline=True), FakeEmbedding)


def test_zhipu_api_key_never_leaks() -> None:
    # 刻意不用 sk- 形状的假值：那会与 scripts/check_secrets.py 的通用规则撞车，
    # 让「密钥扫描」这道闸门长期误报。本测试要验的是「值不外泄」，与形状无关。
    secret = "unit-test-fixture-value-that-must-not-leak"
    cfg = base_settings.model_copy(update={"zhipu_api_key": SecretStr(secret)})
    model = ZhipuEmbedding(cfg=cfg)
    assert secret not in repr(model)
    assert secret not in model.model_dump_json()
    assert secret not in str(model.model_dump())
    assert model.dimensions == cfg.embedding_dimensions


def test_zhipu_rejects_unsupported_dimension() -> None:
    cfg = base_settings.model_copy(
        update={"zhipu_api_key": SecretStr("k"), "embedding_dimensions": 777}
    )
    with pytest.raises(PermanentExternalError):
        ZhipuEmbedding(cfg=cfg)


# --------------------------------------------------------------------------- #
# 抽取器与图 schema
# --------------------------------------------------------------------------- #


def test_structural_refs_reads_identifiers_out_of_free_text() -> None:
    segments, frames = structural_refs("因设计公司修改污水井位置 301分段FR36污水井处更换加厚板")
    assert segments == ["301分段"]
    assert frames == ["FR36"]


def _extract(metadata: dict, text: str):
    node = TextNode(text=text, metadata=metadata)
    return DomainTripletExtractor()([node])[0]


def test_extractor_emits_llamaindex_kg_metadata() -> None:
    node = _extract(
        {
            "case_id": "H-02-1",
            "group_key": "H-02",
            "component": "污水井",
            "departments": ["船体车间", "质保部"],
            "reason": "设计公司修改",
            "time_point": "施工前修改",
        },
        "301分段FR36污水井处更换加厚板",
    )
    triples = triples_from_nodes([node])
    kinds = {(t.subject_label, t.relation, t.object_label) for t in triples}
    assert ("CHANGE_ORDER", "MODIFIES", "COMPONENT") in kinds
    assert ("CHANGE_ORDER", "SIGNED_BY", "DEPARTMENT") in kinds
    assert ("CHANGE_ORDER", "HAS_REASON", "REASON") in kinds
    assert ("CHANGE_ORDER", "OCCURS_AT", "TIME_POINT") in kinds
    assert ("COMPONENT", "PART_OF", "COMPONENT") in kinds
    # 所有三元组都必须能过 schema 校验
    assert all(validate_triple(t) is None for t in triples)


def test_part_of_only_when_the_parent_is_unambiguous() -> None:
    """两个分段都出现时不猜归属：猜错会让 expand() 推出错误的专业集。"""
    one = _extract({"case_id": "A-1", "component": "污水井"}, "301分段更换加厚板")
    assert any(t.relation == "PART_OF" for t in triples_from_nodes([one]))

    two = _extract({"case_id": "A-2", "component": "污水井"}, "301分段与302分段都要改")
    assert not any(t.relation == "PART_OF" for t in triples_from_nodes([two]))


def test_extractor_skips_nodes_without_case_id() -> None:
    node = _extract({"component": "污水井"}, "无单号")
    assert triples_from_nodes([node]) == []


@pytest.mark.parametrize(
    ("triple", "why"),
    [
        (Triple("COMPONENT", "A", "SIGNED_BY", "DEPARTMENT", "B"), "COMPONENT 不允许 SIGNED_BY"),
        (Triple("CHANGE_ORDER", "A", "MODIFIES", "DEPARTMENT", "B"), "端点类型不匹配"),
        (Triple("CHANGE_ORDER", "A", "INVENTED", "COMPONENT", "B"), "未知关系"),
        (Triple("CHANGE_ORDER", "A", "HAS_REASON", "REASON", "A"), "自环"),
    ],
)
def test_validate_triple_rejects_off_schema_edges(triple: Triple, why: str) -> None:
    assert validate_triple(triple) is not None, why


def test_validate_triple_accepts_the_domain_schema() -> None:
    assert validate_triple(Triple("CHANGE_ORDER", "H-01", "MODIFIES", "COMPONENT", "污水井")) is None


def test_kg_extractor_selection() -> None:
    """``rule`` 是默认；``llm`` 走 LlamaIndex 的 SchemaLLMPathExtractor。

    这里验证**接线**是通的（Literal 类型、合法关系表、CustomLLM 桥接都能构造）。
    抽取质量本身需要真实 key，属未验证项，见 docs/rag.md §9。
    """
    from ec_renew.rag_llama.extractors import build_kg_extractor

    assert isinstance(build_kg_extractor(mode="rule"), DomainTripletExtractor)
    with pytest.raises(ValueError):
        build_kg_extractor(mode="nonsense")
    # offline=True 时用 FakeLLM 桥接；只要能构造出来就说明 schema 是合法的。
    assert build_kg_extractor(mode="llm", offline=True) is not None


# --------------------------------------------------------------------------- #
# 融合排序
# --------------------------------------------------------------------------- #


def test_rrf_fuse_is_rank_based_and_deterministic() -> None:
    left = ["A", "B", "C"]
    right = ["B", "A", "D"]
    fused = rrf_fuse([left, right], k=60)
    assert fused == rrf_fuse([left, right], k=60)
    # A 与 B 都在两条腿的前二；D 只被一条腿排到第三
    assert fused["A"] > fused["D"]
    assert fused["B"] > fused["D"]
    # 融合只看排名，不看分数：同样的排名集合，顺序不影响结果
    assert rrf_fuse([left, right], k=60) == rrf_fuse([right, left], k=60)


def test_rrf_fuse_ignores_empty_keys() -> None:
    fused = rrf_fuse([["A", ""], []], k=60)
    assert len(fused) == 1
    assert fused["A"] == pytest.approx(1 / 61)


def test_rank_graph_rows_prefers_more_term_matches() -> None:
    rows = [
        {"case_id": "H-01", "components": ["缆绳"], "reasons": [], "timepoints": [], "departments": []},
        {
            "case_id": "H-02-1",
            "components": ["污水井"],
            "reasons": ["设计公司修改"],
            "timepoints": [],
            "departments": [],
        },
    ]
    ranked = rank_graph_rows(rows, ["污水井", "设计公司修改"])
    assert ranked[0] == "H-02-1"
    # 同分时按 case_id 排序 —— 重放结果必须一致
    assert rank_graph_rows(list(reversed(rows)), ["污水井", "设计公司修改"]) == ranked


# --------------------------------------------------------------------------- #
# 集成：真摄取 + 真检索（服务不可用则 skip）
# --------------------------------------------------------------------------- #


def _scratch_settings():
    return base_settings.model_copy(
        update={"qdrant_collection": SCRATCH_COLLECTION, "embedding_dimensions": 256}
    )


def _sample_settings():
    """把摄取指向合成语料：无语料环境下也能跑通整条链路。"""
    return base_settings.model_copy(
        update={
            "qdrant_collection": SAMPLE_COLLECTION,
            "embedding_dimensions": 256,
            "data_dir": SAMPLE_CORPUS.parent,
            "corpus_file": SAMPLE_CORPUS.name,
        }
    )


def _drop_collection(cfg) -> None:
    from ec_renew.rag_llama.vector_store import build_client

    client = build_client(cfg)
    if client.collection_exists(cfg.qdrant_collection):
        client.delete_collection(cfg.qdrant_collection)


def _purge_synthetic_graph(cfg) -> None:
    """回收合成语料写进 Neo4j 的节点。

    合成单号统一用 ``S-`` 前缀正是为了这一步：Neo4j 社区版只有一个 database，
    集成测试与真实数据同库，只有靠可识别的前缀才能干净收回。
    """
    from ec_renew.rag_llama.graph_store import Neo4jDomainWriter

    writer = Neo4jDomainWriter(cfg)
    try:
        writer.purge_change_orders([SYNTHETIC_PREFIX])
    finally:
        writer.close()


def _stack_available() -> bool:
    cfg = _scratch_settings()
    try:
        from ec_renew.rag_llama.vector_store import build_client, ping

        ping(build_client(cfg))
        from ec_renew.rag_llama.graph_store import Neo4jDomainWriter

        writer = Neo4jDomainWriter(cfg)
        try:
            writer.ping()
        finally:
            writer.close()
    except Exception:  # noqa: BLE001 — 探测用：任何失败都等同于「服务不可用」
        return False
    return True


requires_stack = pytest.mark.skipif(
    not _stack_available(), reason="需要 docker 部署的 Qdrant + Neo4j"
)


@pytest.fixture(scope="module")
def ingested_collection():
    # 真实语料不进版本库 —— 缺了就 skip，而不是把「没有数据」伪装成失败。
    if not has_real_corpus:
        pytest.skip(f"缺少真实语料 {REAL_CORPUS}（见 data/README.md）")

    cfg = _scratch_settings()
    from ec_renew.rag_llama.ingest import ingest_sync
    from ec_renew.rag_llama.vector_store import build_client

    # 断言写在 WriteReport（本次写了什么）而不是 counts（库里总共有什么）：
    # 后者依赖其它数据是否存在，会让用例互相依赖、顺序敏感。
    report = ingest_sync(cfg, offline=True, recreate=True)
    assert report.vector["points"] == 118
    assert report.graph["nodes"]["CHANGE_ORDER"] == 118

    # 幂等：再摄一次，写入量与点数都不应变化。
    again = ingest_sync(cfg, offline=True)
    assert again.vector["points"] == 118
    assert again.graph["relations"]["SIGNED_BY"] == 728

    yield cfg

    client = build_client(cfg)
    client.delete_collection(SCRATCH_COLLECTION)


@pytest.fixture(scope="module")
def sample_collection():
    """合成语料的完整摄取 —— CI 无业务数据时的主要端到端覆盖。"""
    cfg = _sample_settings()
    from ec_renew.rag_llama.ingest import ingest_sync

    report = ingest_sync(cfg, offline=True, recreate=True)
    assert report.vector["points"] == 8
    assert report.graph["nodes"]["CHANGE_ORDER"] == 8
    # 合成语料 8 张单覆盖 6 个变更组（S-02 / S-03 各含 2 张子单）
    assert len({o.group_key for o in load_change_orders(SAMPLE_CORPUS)}) == 6

    # 幂等
    again = ingest_sync(cfg, offline=True)
    assert again.vector["points"] == 8

    try:
        yield cfg
    finally:
        # 图侧必须回收：Neo4j 社区版只有一个 database，合成数据与真实数据同库。
        # 放在 finally 里，断言失败时也会执行，否则一次失败会永久污染真实图。
        _purge_synthetic_graph(cfg)
        _drop_collection(cfg)


@requires_stack
def test_pipeline_end_to_end_on_synthetic_corpus(sample_collection) -> None:
    """不依赖业务数据，也要能证明「摄取 → 建图 → 检索」是通的。"""
    from ec_renew.rag_llama.retriever import LlamaIndexRetriever

    retriever = LlamaIndexRetriever(sample_collection, offline=True)
    try:
        assert retriever.health()["points"] == 8

        async def scenario():
            request = "301分段FR36污水井更换加厚板，涉及焊接"
            entities = await retriever.link(request)
            names = [e.name for e in entities]
            assert "污水井" in names
            assert "FR36" in names
            assert "FR3" not in names, "FR36 不得把 FR3 也带出来"

            expansion = await retriever.expand(entities)
            assert expansion.historical_departments
            assert expansion.historical_disciplines

            return await retriever.prefetch([request, "污水井"], 4)

        bundle = asyncio.run(scenario())
        assert bundle.items, "共享事实基线不能为空"
        assert bundle.items[0].source == "graph"
        assert bundle.items[0].content.startswith("单号:")
        assert bundle.baseline_ids == [item.evidence_id for item in bundle.items]
    finally:
        retriever.close()


@requires_stack
def test_hybrid_retriever_end_to_end(ingested_collection) -> None:
    from ec_renew.rag_llama.retriever import LlamaIndexRetriever

    retriever = LlamaIndexRetriever(ingested_collection, offline=True)
    try:
        health = retriever.health()
        assert health["points"] == 118

        async def scenario():
            request = "301分段FR36污水井更换加厚板，涉及焊接，需确认合规性"
            entities = await retriever.link(request)
            names = [e.name for e in entities]
            assert "污水井" in names
            assert "FR36" in names
            assert "FR3" not in names, "FR36 不得把 FR3 也带出来"

            expansion = await retriever.expand(entities)
            assert expansion.historical_departments
            assert expansion.historical_disciplines
            assert "301分段" in expansion.parent_components

            bundle = await retriever.prefetch([request, "污水井"], 6)
            return bundle

        bundle = asyncio.run(scenario())
        assert bundle.items, "共享事实基线不能为空"
        # baseline 必须由 items 推出（EvidenceBundle 的校验器保证）
        assert bundle.baseline_ids == [item.evidence_id for item in bundle.items]
        # 命中的污水井历史单必须排在最前，且被判定为真实历史案例
        assert bundle.items[0].evidence_id == "CASE-H-02-1"
        assert bundle.items[0].source == "graph"
        assert bundle.items[0].disciplines
        # 证据正文是语料原文，不是拼出来的摘要
        assert bundle.items[0].content.startswith("单号:H-02-1")
        # 纯向量命中不计入历史依据，但必须可见
        assert any(w.startswith("vector_only_hits") for w in bundle.warnings)
    finally:
        retriever.close()


@requires_stack
def test_prefetch_is_deterministic(ingested_collection) -> None:
    from ec_renew.rag_llama.retriever import LlamaIndexRetriever

    def once():
        retriever = LlamaIndexRetriever(ingested_collection, offline=True)
        try:
            return asyncio.run(retriever.prefetch(["污水井 加厚板"], 5))
        finally:
            retriever.close()

    first, second = once(), once()
    assert [i.evidence_id for i in first.items] == [i.evidence_id for i in second.items]
    assert [round(i.score, 6) for i in first.items] == [round(i.score, 6) for i in second.items]


@requires_stack
def test_factory_auto_selects_llamaindex_when_healthy(ingested_collection) -> None:
    from ec_renew.rag_llama.factory import build_retriever
    from ec_renew.rag_llama.retriever import LlamaIndexRetriever

    retriever, notes = build_retriever(ingested_collection, mode="auto", offline=True)
    try:
        assert isinstance(retriever, LlamaIndexRetriever)
        assert len(notes) == 1, f"健康时不应出现降级说明：{notes}"
    finally:
        retriever.close()


def test_factory_explicit_backend_never_silently_degrades() -> None:
    """显式要求某后端时不许偷偷换掉 —— 否则「查不到」会被伪装成「没有」。"""
    from ec_renew.errors import InvalidRequest
    from ec_renew.rag_llama.factory import build_retriever

    cfg = base_settings.model_copy(update={"zhipu_api_key": SecretStr("")})
    with pytest.raises(InvalidRequest):
        build_retriever(cfg, mode="llamaindex")
    with pytest.raises(InvalidRequest):
        build_retriever(cfg, mode="nonsense")


def test_factory_memory_needs_no_services() -> None:
    from ec_renew.rag import InMemoryRetriever
    from ec_renew.rag_llama.factory import build_retriever

    retriever, notes = build_retriever(base_settings, mode="memory")
    assert isinstance(retriever, InMemoryRetriever)
    assert len(notes) == 1


# --------------------------------------------------------------------------- #
# 智谱 API 契约（用 MockTransport，不需要真实 key）
#
# 这是「没有 key 也能验证真实链路」的关键：请求长什么样、返回乱序怎么办、
# 各类 HTTP 错误翻成哪个异常，全都在这里钉死。
# --------------------------------------------------------------------------- #


def _zhipu_with_transport(handler, **updates) -> ZhipuEmbedding:
    cfg = base_settings.model_copy(
        update={"zhipu_api_key": SecretStr("test-key"), **updates}
    )
    model = ZhipuEmbedding(cfg=cfg)
    # 直接替换同步 client：把网络换成 MockTransport，其余逻辑一行不变。
    model._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=cfg.zhipu_base_url,
        headers=model._headers(),
    )
    return model


def test_zhipu_request_shape_and_batch_order() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        # 故意倒序返回：不按 index 重排就会把向量和文本错配。
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [2.0, 2.0]},
                    {"index": 0, "embedding": [1.0, 1.0]},
                ],
                "usage": {"total_tokens": 7},
            },
        )

    model = _zhipu_with_transport(handler, embedding_dimensions=2048)
    vectors = model.get_text_embedding_batch(["甲", "乙"])

    assert seen["url"].endswith("/embeddings")
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "embedding-3"
    assert seen["body"]["dimensions"] == 2048
    assert seen["body"]["input"] == ["甲", "乙"]
    assert vectors == [[1.0, 1.0], [2.0, 2.0]]


@pytest.mark.parametrize(
    ("status", "expected"),
    [(400, PermanentExternalError), (500, TransientError)],
)
def test_zhipu_translates_http_errors(status: int, expected: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "boom"})

    model = _zhipu_with_transport(handler)
    model.max_attempts = 1  # 不重试，让测试只验证「翻译」这一件事
    with pytest.raises(expected):
        model.get_text_embedding("x")


def test_zhipu_retries_transient_then_succeeds() -> None:
    """429/5xx 必须重试；不重试会把一次限流直接变成一次失败。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    model = _zhipu_with_transport(handler)
    assert model.get_text_embedding("x") == [1.0]
    assert calls["n"] == 2


def test_zhipu_rate_limited_after_exhausting_attempts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "0"}, json={})

    model = _zhipu_with_transport(handler)
    with pytest.raises(RateLimited):
        model.get_text_embedding("x")


def test_zhipu_rejects_malformed_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    model = _zhipu_with_transport(handler)
    model.max_attempts = 1
    with pytest.raises(PermanentExternalError):
        model.get_text_embedding("x")


def test_zhipu_async_path_matches_sync() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0]}]})

    model = _zhipu_with_transport(handler)
    model._aclient = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=model.api_base,
        headers=model._headers(),
    )
    assert asyncio.run(model.aget_query_embedding("x")) == [3.0]
