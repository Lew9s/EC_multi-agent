"""Neo4j 图存储（本项目领域 schema）。

只做三件事：**建约束**、**写三元组**、**清空**。读取在 ``rag/graph.py``
（Cypher 只允许出现在这两个文件里）。

设计取舍：写入用**静态 Cypher + UNWIND 批量**，不用 APOC 的动态 label。
``apoc.merge.node`` 会把 label 拼进查询串，既是注入面，也让「合法 schema」
这件事失去编译期约束；而领域关系只有 5 种、端点组合固定，写死反而更安全、
更快（一次往返写一批），部署端也不需要装插件。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from ..config import Settings
from ..config import settings as default_settings
from ..errors import PermanentExternalError, TransientError
from .extractors import DOMAIN_ENTITIES, DOMAIN_RELATIONS, KG_VALIDATION_SCHEMA

# --------------------------------------------------------------------------- #
# Schema — labels are literals, so no query string is ever built from data
# --------------------------------------------------------------------------- #

_NODE_UPSERT: dict[str, str] = {
    "CHANGE_ORDER": (
        "UNWIND $rows AS row "
        "MERGE (n:CHANGE_ORDER {name: row.name}) "
        "SET n.group_key = row.group_key"
    ),
    "COMPONENT": "UNWIND $rows AS row MERGE (n:COMPONENT {name: row.name})",
    "DEPARTMENT": "UNWIND $rows AS row MERGE (n:DEPARTMENT {name: row.name})",
    "REASON": "UNWIND $rows AS row MERGE (n:REASON {name: row.name})",
    "TIME_POINT": "UNWIND $rows AS row MERGE (n:TIME_POINT {name: row.name})",
}

# relation -> (source label, target label). The pair is the schema: a relation
# that does not appear here is rejected before it reaches the database.
_RELATION_ENDS: dict[str, tuple[str, str]] = {
    "MODIFIES": ("CHANGE_ORDER", "COMPONENT"),
    "SIGNED_BY": ("CHANGE_ORDER", "DEPARTMENT"),
    "HAS_REASON": ("CHANGE_ORDER", "REASON"),
    "OCCURS_AT": ("CHANGE_ORDER", "TIME_POINT"),
    "PART_OF": ("COMPONENT", "COMPONENT"),
}

_REL_UPSERT: dict[str, str] = {
    relation: (
        f"UNWIND $rows AS row "
        f"MATCH (a:{source} {{name: row.source}}) "
        f"MATCH (b:{target} {{name: row.target}}) "
        f"MERGE (a)-[:{relation}]->(b)"
    )
    for relation, (source, target) in _RELATION_ENDS.items()
}

_CONSTRAINTS: tuple[str, ...] = tuple(
    f"CREATE CONSTRAINT ec_{label.lower()}_name IF NOT EXISTS "
    f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
    for label in DOMAIN_ENTITIES
)

# 清空只针对本项目的领域 label：不碰同一库里其它 schema 的数据。
_RESET_DOMAIN = (
    "MATCH (n) WHERE " + " OR ".join(f"n:{label}" for label in DOMAIN_ENTITIES) + " DETACH DELETE n"
)
_WIPE_ALL = "MATCH (n) DETACH DELETE n"

# 按单号前缀回收：用于清掉一次实验/测试写入的合成数据，而不动其它数据。
# DETACH DELETE 不能直接 RETURN，所以先 collect 再 FOREACH。
_PURGE_CHANGE_ORDERS = """
MATCH (c:CHANGE_ORDER)
WHERE any(prefix IN $prefixes WHERE c.name STARTS WITH prefix)
WITH collect(c) AS victims
FOREACH (victim IN victims | DETACH DELETE victim)
RETURN size(victims) AS deleted
"""

# 只删「已经没有任何关系」的领域节点：被真实变更单共享的节点会保留下来。
_PURGE_ORPHANS = """
MATCH (n)
WHERE (n:COMPONENT OR n:REASON OR n:DEPARTMENT OR n:TIME_POINT) AND NOT (n)--()
WITH collect(n) AS victims
FOREACH (victim IN victims | DELETE victim)
RETURN size(victims) AS deleted
"""

_COUNT_ALL = (
    "MATCH (n) WHERE "
    + " OR ".join(f"n:{label}" for label in DOMAIN_ENTITIES)
    + " UNWIND labels(n) AS label RETURN label, count(*) AS count"
)
_COUNT_RELATIONS = "MATCH ()-[r]->() WHERE type(r) IN $types RETURN type(r) AS type, count(*) AS count"


@dataclass
class WriteReport:
    """写入统计。确定性，可用于比对两次摄取是否一致。"""

    nodes: dict[str, int] = field(default_factory=dict)
    relations: dict[str, int] = field(default_factory=dict)

    @property
    def total_nodes(self) -> int:
        return sum(self.nodes.values())

    @property
    def total_relations(self) -> int:
        return sum(self.relations.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": dict(sorted(self.nodes.items())),
            "relations": dict(sorted(self.relations.items())),
            "total_nodes": self.total_nodes,
            "total_relations": self.total_relations,
        }


@dataclass(frozen=True)
class Triple:
    """领域三元组（已通过 schema 校验）。"""

    subject_label: str
    subject: str
    relation: str
    object_label: str
    object: str
    subject_properties: dict[str, Any] = field(default_factory=dict)


def validate_triple(triple: Triple) -> str | None:
    """返回拒绝原因；``None`` 表示合法。

    宁可丢掉一条编造的边，也不要让它污染 ``expand()`` 推出的专业归属 ——
    下游的专家激活完全建立在这个图上。
    """
    if triple.relation not in DOMAIN_RELATIONS:
        return f"未知关系 {triple.relation!r}"
    if triple.subject_label not in DOMAIN_ENTITIES:
        return f"未知实体类型 {triple.subject_label!r}"
    if triple.object_label not in DOMAIN_ENTITIES:
        return f"未知实体类型 {triple.object_label!r}"
    expected = _RELATION_ENDS[triple.relation]
    if (triple.subject_label, triple.object_label) != expected:
        return (
            f"关系 {triple.relation!r} 要求 {expected[0]}->{expected[1]}，"
            f"实际 {triple.subject_label}->{triple.object_label}"
        )
    allowed = KG_VALIDATION_SCHEMA.get(triple.subject_label, [])
    if triple.relation not in allowed:
        return f"{triple.subject_label} 不允许 {triple.relation!r}"
    if not triple.subject or not triple.object:
        return "端点名为空"
    if triple.subject == triple.object and triple.relation != "PART_OF":
        return "自环"
    return None


class Neo4jDomainWriter:
    """领域图的唯一写入口。"""

    def __init__(self, cfg: Settings | None = None, *, database: str = "neo4j") -> None:
        cfg = cfg or default_settings
        self._database = database
        try:
            self._driver = GraphDatabase.driver(
                cfg.neo4j_uri,
                auth=(cfg.neo4j_user, cfg.neo4j_password.get_secret_value()),
            )
        except Exception as exc:  # neo4j.* -> our hierarchy
            raise PermanentExternalError(f"Neo4j 驱动初始化失败: {exc}") from exc

    def close(self) -> None:
        self._driver.close()

    # -- exception translation (the only place neo4j.* is visible here) ---- #
    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        try:
            with self._driver.session(database=self._database) as session:
                return [dict(record) for record in session.run(query, **params)]
        except Neo4jError as exc:
            name = type(exc).__name__
            if "ServiceUnavailable" in name or "SessionExpired" in name:
                raise TransientError(f"Neo4j 不可用: {exc}") from exc
            raise PermanentExternalError(f"Neo4j 写入失败: {exc}") from exc
        except OSError as exc:
            raise TransientError(f"Neo4j 连接失败: {exc}") from exc

    # -- lifecycle ---------------------------------------------------------- #
    def ping(self) -> bool:
        self._run("RETURN 1 AS ok")
        return True

    def ensure_schema(self) -> None:
        for statement in _CONSTRAINTS:
            self._run(statement)

    def reset_domain(self) -> None:
        """只删本项目的领域节点，保留库里其它 schema。"""
        self._run(_RESET_DOMAIN)

    def wipe_all(self) -> None:
        """谨慎：清空整个库（用于一次性迁移，例如清掉历史遗留的旧 schema）。"""
        self._run(_WIPE_ALL)

    def purge_change_orders(self, prefixes: Sequence[str]) -> dict[str, int]:
        """按单号前缀删除变更单，并回收因此变成孤立的领域节点。

        集成测试要往同一个库里写合成语料（Neo4j 社区版只有一个 database），
        跑完必须能原样收回；生产上也可用于撤销一次错误的批量摄取。
        """
        prefixes = [p for p in prefixes if p]
        if not prefixes:
            return {"CHANGE_ORDER": 0, "orphans": 0}
        removed = self._run(_PURGE_CHANGE_ORDERS, prefixes=prefixes)
        orphans = self._run(_PURGE_ORPHANS)
        return {
            "CHANGE_ORDER": int(removed[0]["deleted"]) if removed else 0,
            "orphans": int(orphans[0]["deleted"]) if orphans else 0,
        }

    # -- write -------------------------------------------------------------- #
    def write_triples(self, triples: Iterable[Triple]) -> WriteReport:
        report = WriteReport()
        nodes: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        relations: dict[str, set[tuple[str, str]]] = defaultdict(set)
        rejected: list[str] = []

        for triple in triples:
            reason = validate_triple(triple)
            if reason is not None:
                rejected.append(f"{triple.relation}({triple.subject}->{triple.object}): {reason}")
                continue
            nodes[triple.subject_label][triple.subject] = _clean(triple.subject_properties)
            nodes[triple.object_label].setdefault(triple.object, {})
            # CHANGE_ORDER 的 group_key 挂在主体属性上。
            if triple.subject_label == "CHANGE_ORDER":
                nodes["CHANGE_ORDER"][triple.subject].setdefault(
                    "group_key", triple.subject_properties.get("group_key", triple.subject)
                )
            relations[triple.relation].add((triple.subject, triple.object))

        for label, by_name in nodes.items():
            rows = [{"name": name, **props} for name, props in sorted(by_name.items())]
            if label == "CHANGE_ORDER":
                for row in rows:
                    row.setdefault("group_key", row["name"])
            if not rows:
                continue
            self._run(_NODE_UPSERT[label], rows=rows)
            report.nodes[label] = len(rows)

        for relation, pairs in relations.items():
            rows = [{"source": source, "target": target} for source, target in sorted(pairs)]
            if not rows:
                continue
            self._run(_REL_UPSERT[relation], rows=rows)
            report.relations[relation] = len(rows)

        if rejected:
            report.nodes["__rejected__"] = len(rejected)
        return report

    # -- read (verification only) ------------------------------------------- #
    def counts(self) -> dict[str, Any]:
        node_rows = self._run(_COUNT_ALL)
        rel_rows = self._run(_COUNT_RELATIONS, types=list(DOMAIN_RELATIONS))
        # 同一节点有多个领域 label 时会重复计数，这里按 label 去重累加即可
        # （本项目每个节点只有一个领域 label）。
        node_counts: dict[str, int] = defaultdict(int)
        for row in node_rows:
            node_counts[str(row["label"])] += int(row["count"])
        return {
            "nodes": dict(sorted(node_counts.items())),
            "relations": {str(r["type"]): int(r["count"]) for r in rel_rows},
        }


def _clean(properties: dict[str, Any]) -> dict[str, Any]:
    """Neo4j 不能把 None 作为属性值写入；顺手丢掉空串。"""
    return {
        key: value
        for key, value in (properties or {}).items()
        if value is not None and value != "" and not isinstance(value, (list, dict, tuple))
    }


def triples_from_nodes(nodes: Sequence[Any]) -> list[Triple]:
    """把带 ``kg_nodes`` / ``kg_relations`` 元数据的节点翻成领域三元组。

    这里刻意只认 LlamaIndex 的标准 KG 元数据键，因此规则抽取器与
    ``SchemaLLMPathExtractor`` 的产物都能喂进来。
    """
    from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY

    triples: list[Triple] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for node in nodes:
        metadata = node.metadata or {}
        entity_nodes = metadata.get(KG_NODES_KEY) or []
        relations = metadata.get(KG_RELATIONS_KEY) or []
        name_of: dict[str, str] = {}
        label_of: dict[str, str] = {}
        props_of: dict[str, dict[str, Any]] = {}
        for entity in entity_nodes:
            name_of[entity.id] = entity.name
            label_of[entity.id] = entity.label
            label_of[entity.name] = entity.label
            props_of[entity.name] = dict(getattr(entity, "properties", {}) or {})
        for relation in relations:
            subject = name_of.get(relation.source_id, relation.source_id)
            obj = name_of.get(relation.target_id, relation.target_id)
            subject_label = label_of.get(relation.source_id, "")
            object_label = label_of.get(relation.target_id, "")
            if not subject_label or not object_label:
                continue
            key = (subject_label, subject, relation.label, object_label, obj)
            if key in seen:
                continue
            seen.add(key)
            triples.append(
                Triple(
                    subject_label=subject_label,
                    subject=subject,
                    relation=relation.label,
                    object_label=object_label,
                    object=obj,
                    subject_properties=props_of.get(subject, {}),
                )
            )
    return triples
