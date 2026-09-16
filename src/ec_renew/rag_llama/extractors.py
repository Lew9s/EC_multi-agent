"""领域三元组抽取（LlamaIndex ``TransformComponent``）。

**为什么是规则而不是 LLM**：本语料的字段本来就带显式标签
（``单号/变更原因/变更时间点/变更对象/签收部门``）。用 LLM 把已有标签的内容
再猜一遍，只会换来不确定性与 token 账单（design.md P7「可复现」）。
真正需要语义理解的是「变更内容」里的结构引用（``301分段`` / ``FR36``），
那部分用保守的正则补抽，规则简单且可解释。

**为什么仍然写成 LlamaIndex 组件**：输出沿用 LlamaIndex 自己的
``KG_NODES_KEY`` / ``KG_RELATIONS_KEY``。这意味着当语料换成没有字段标签的
自由文本时，把 ``SchemaLLMPathExtractor`` 直接换进同一条 ``IngestionPipeline``
即可，下游写库代码一行都不用改。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    EntityNode,
    Relation,
)
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from pydantic import Field

# 领域 schema（design.md §0.3）—— 与 rag.py 的 Cypher 严格一致。
DOMAIN_ENTITIES: tuple[str, ...] = (
    "CHANGE_ORDER",
    "COMPONENT",
    "DEPARTMENT",
    "REASON",
    "TIME_POINT",
)
DOMAIN_RELATIONS: tuple[str, ...] = (
    "MODIFIES",
    "SIGNED_BY",
    "HAS_REASON",
    "OCCURS_AT",
    "PART_OF",
)

# 合法关系约束，交给 LlamaIndex 的抽取器做 strict 校验时复用。
KG_VALIDATION_SCHEMA: dict[str, list[str]] = {
    "CHANGE_ORDER": ["MODIFIES", "SIGNED_BY", "HAS_REASON", "OCCURS_AT"],
    "COMPONENT": ["MODIFIES", "PART_OF"],
    "DEPARTMENT": ["SIGNED_BY"],
    "REASON": ["HAS_REASON"],
    "TIME_POINT": ["OCCURS_AT"],
}

_COMPONENT_SPLIT_RE = re.compile(r"[，,、;；]+")
_SEGMENT_RE = re.compile(r"(\d{3,4})\s*分段")
_FRAME_RE = re.compile(r"FR\s*(\d+)", re.IGNORECASE)

# 单张变更单最多补抽多少个结构引用：防止正文里的编号把图撑爆。
MAX_STRUCTURAL_REFS = 6


def _split_components(value: str) -> list[str]:
    return [part.strip() for part in _COMPONENT_SPLIT_RE.split(value or "") if part.strip()]


def structural_refs(text: str, limit: int = MAX_STRUCTURAL_REFS) -> tuple[list[str], list[str]]:
    """从「变更内容」里补抽结构引用。

    返回 ``(segments, frames)``，例如 ``(["301分段"], ["FR36"])``。
    编号原样保留（``FR36`` 而不是 ``36``），因为图里的节点名要和用户在
    请求里写的字符串对得上，否则 ``link()`` 的 CONTAINS 匹配会失效。
    """
    segments: list[str] = []
    for match in _SEGMENT_RE.finditer(text or ""):
        name = f"{match.group(1)}分段"
        if name not in segments:
            segments.append(name)
    frames: list[str] = []
    for match in _FRAME_RE.finditer(text or ""):
        name = f"FR{match.group(1)}"
        if name not in frames:
            frames.append(name)
    return segments[:limit], frames[:limit]


class DomainTripletExtractor(TransformComponent):
    """结构化变更单 -> ``kg_nodes`` / ``kg_relations``（领域 schema）。"""

    include_structural: bool = True
    include_parent: bool = True
    max_triplets_per_chunk: int = Field(default=40, ge=1)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        del kwargs
        for node in nodes:
            self._extract_one(node)
        return nodes

    def _extract_one(self, node: BaseNode) -> BaseNode:
        metadata = dict(node.metadata or {})
        case_id = str(metadata.get("case_id") or "").strip()
        if not case_id:
            # 没有单号就没有可回溯的主体，直接跳过而不是编一个 id。
            return node

        group_key = str(metadata.get("group_key") or case_id)
        text = node.get_content() if hasattr(node, "get_content") else ""

        primary = _split_components(str(metadata.get("component") or ""))
        segments: list[str] = []
        frames: list[str] = []
        if self.include_structural:
            segments, frames = structural_refs(text)

        component_names: list[str] = []
        for name in [*primary, *segments, *frames]:
            if name and name not in component_names:
                component_names.append(name)

        entity_nodes: list[EntityNode] = [
            EntityNode(
                label="CHANGE_ORDER",
                name=case_id,
                properties={
                    "group_key": group_key,
                    "source_file": str(metadata.get("source_file") or ""),
                    "order_index": metadata.get("order_index"),
                },
            )
        ]
        relations: list[Relation] = []

        def link(label: str, name: str) -> None:
            entity_nodes.append(EntityNode(label=label, name=name))
            relations.append(
                Relation(
                    label={
                        "COMPONENT": "MODIFIES",
                        "DEPARTMENT": "SIGNED_BY",
                        "REASON": "HAS_REASON",
                        "TIME_POINT": "OCCURS_AT",
                    }[label],
                    source_id=case_id,
                    target_id=name,
                )
            )

        for name in component_names:
            link("COMPONENT", name)
        for name in metadata.get("departments") or []:
            if str(name).strip():
                link("DEPARTMENT", str(name).strip())
        reason = str(metadata.get("reason") or "").strip()
        if reason:
            link("REASON", reason)
        time_point = str(metadata.get("time_point") or "").strip()
        if time_point:
            link("TIME_POINT", time_point)

        # PART_OF 只在一个分段被明确提到时才建：多个分段时归属有歧义，
        # 猜错会让 expand() 的「父结构历史」张冠李戴，宁可缺不可错。
        if self.include_parent and len(segments) == 1:
            parent = segments[0]
            for name in primary:
                if name and name != parent:
                    relations.append(
                        Relation(label="PART_OF", source_id=name, target_id=parent)
                    )

        if len(relations) > self.max_triplets_per_chunk:
            relations = relations[: self.max_triplets_per_chunk]

        node.metadata[KG_NODES_KEY] = entity_nodes
        node.metadata[KG_RELATIONS_KEY] = relations
        return node


def build_kg_extractor(*, mode: str = "rule", offline: bool = False) -> TransformComponent:
    """选抽取器。

    ``rule``（默认）—— 确定性规则，无 key，可复现。
    ``llm``         —— LlamaIndex ``SchemaLLMPathExtractor``，需要 DEEPSEEK_API_KEY；
                       只对**没有字段标签**的自由文本文档有意义。
    """
    if mode == "rule":
        return DomainTripletExtractor()
    if mode != "llm":
        raise ValueError(f"未知的 KG_EXTRACTOR: {mode!r}（可选 rule | llm）")

    from typing import Literal

    from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

    from .llm import build_llama_llm

    entities = Literal[DOMAIN_ENTITIES]  # type: ignore[valid-type]
    relations = Literal[DOMAIN_RELATIONS]  # type: ignore[valid-type]
    return SchemaLLMPathExtractor(
        llm=build_llama_llm(offline=offline, purpose="kg_extract"),
        possible_entities=entities,
        possible_relations=relations,
        kg_validation_schema=KG_VALIDATION_SCHEMA,
        num_workers=1,
        max_triplets_per_chunk=20,
        strict=True,
    )


def discipline_filters(disciplines: Sequence[str]) -> MetadataFilters:
    """按专业过滤的向量检索条件（``disciplines`` 在建库时已打标）。

    用 ``should`` 语义（OR）而不是 AND：一张变更单通常同时涉及多个专业，
    要求「全部包含」会把本该命中的案例排除掉。
    """
    return MetadataFilters(
        filters=[MetadataFilter(key="disciplines", value=list(disciplines))],
        condition="or",
    )
