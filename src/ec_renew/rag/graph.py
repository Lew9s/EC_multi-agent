"""Retrievers.

``InMemoryRetriever``  — offline fixture, lets the whole pipeline run with no
                        Neo4j (used by the smoke test and ``--offline``).
``Neo4jRetriever``     — the real one. All Cypher lives here and nowhere else;
                        the driver is sync, so every call is pushed to a worker
                        thread. Third-party exceptions are translated here.

Department -> discipline mapping is a static table for the demo (Q-01 option
"tag at build time" is deferred; a lookup table is enough to prove the flow).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Sequence
from typing import Any

from neo4j import READ_ACCESS, GraphDatabase

from ..config import Settings
from ..contracts import (
    EntityRef,
    EvidenceBundle,
    EvidenceMeta,
    GraphExpansion,
)
from ..errors import PermanentExternalError, TransientError
from .cypher_guard import validate_read_only

# --------------------------------------------------------------------------- #
# Static mapping (demo shortcut)
# --------------------------------------------------------------------------- #

DEPT_TO_DISCIPLINE: dict[str, str] = {
    # 结构
    "船体车间": "E01",
    "放样室": "E01",
    # 舾装 / 工艺 / 生产准备
    "舾冷车间": "E02",
    "生产部": "E02",
    "准备车间": "E02",
    "定额室": "E02",
    "仓储科": "E02",
    # 轮机 / 管系
    "机装车间": "E05",
    "管装车间": "E05",
    "轮机车间": "E05",
    # 电气
    "电装车间": "E04",
    # 质量
    "质保部": "E03",
    "质检部": "E03",
    # 材料与焊接
    "物供部": "E06",
}

DEFAULT_DISCIPLINE = "E03"  # 任何变更都涉及规范/质量


def disciplines_for_departments(departments: Iterable[str]) -> list[str]:
    found = {DEPT_TO_DISCIPLINE.get(d, DEFAULT_DISCIPLINE) for d in departments if d}
    return sorted(found)


def is_partial_identifier(name: str, request: str) -> bool:
    """``name`` 在 ``request`` 里是否只是某个更长编号的前缀。

    ``link()`` 走的是 ``$request CONTAINS c.name``：请求里写着 ``FR36`` 时，
    图里的 ``FR3`` 也会命中，于是意图补全凭空多出一个不存在的实体，
    后续 ``expand()`` 会推出错误的专业集。判据是「以数字结尾的名字，
    在请求里每次出现后面都还跟着数字」—— 只有全部出现都被截断才算部分匹配，
    所以 ``FR3 与 FR36`` 同时出现时 ``FR3`` 依然保留。
    """
    if not name or not name[-1].isdigit():
        return False
    return re.search(re.escape(name) + r"(?![0-9])", request) is None


# --------------------------------------------------------------------------- #
# Offline fixture
# --------------------------------------------------------------------------- #


class InMemoryRetriever:
    """Deterministic fixture: enough shape to exercise every code path."""

    def __init__(self, components: Sequence[str] | None = None, cases: int = 3) -> None:
        self.components = list(components or ["FR36", "污水井", "肋板"])
        self.cases = cases

    async def link(self, request: str) -> list[EntityRef]:
        return [
            EntityRef(name=name, kind="COMPONENT", in_graph=True, graph_key=name)
            for name in self.components
            if name in request
        ]

    async def expand(self, entities: list[EntityRef]) -> GraphExpansion:
        if not entities:
            return GraphExpansion()
        return GraphExpansion(
            parent_components=["301分段"],
            historical_departments=["船体车间", "质保部", "准备车间"],
            historical_disciplines=["E01", "E03", "E06"],
            similar_case_ids=[f"H-{i:02d}-1" for i in range(1, self.cases + 1)],
        )

    async def prefetch(self, queries: list[str], top_k: int) -> EvidenceBundle:
        items: list[EvidenceMeta] = []
        for index in range(1, self.cases + 1):
            items.append(
                EvidenceMeta(
                    evidence_id=f"CASE-{index:03d}",
                    entity_kinds=["CHANGE_ORDER"],
                    entity_keys=[f"H-{index:02d}-1"],
                    disciplines=["E01", "E03"],
                    group_keys=[f"H-{index:02d}"],
                    source="graph",
                    score=0.9 - index * 0.05,
                    content=(
                        f"H-{index:02d}-1：因设计公司修改，施工前更换加厚板；"
                        f"变更对象 污水井；签收部门 船体车间、质保部"
                    ),
                )
            )
        return EvidenceBundle(
            round=0,
            baseline_ids=[item.evidence_id for item in items],
            items=items[:top_k],
            warnings=[],
        )


# --------------------------------------------------------------------------- #
# Neo4j
# --------------------------------------------------------------------------- #

_CYPHER_LINK = """
MATCH (c:COMPONENT)
WHERE $request CONTAINS c.name
RETURN DISTINCT c.name AS name
LIMIT $limit
"""

_CYPHER_EXPAND = """
UNWIND $names AS name
MATCH (comp:COMPONENT {name: name})
OPTIONAL MATCH (comp)-[:PART_OF]->(parent:COMPONENT)
OPTIONAL MATCH (co:CHANGE_ORDER)-[:MODIFIES]->(comp)
OPTIONAL MATCH (co)-[:SIGNED_BY]->(dept:DEPARTMENT)
OPTIONAL MATCH (sib:COMPONENT)-[:PART_OF]->(parent)
OPTIONAL MATCH (co2:CHANGE_ORDER)-[:MODIFIES]->(sib)
OPTIONAL MATCH (co2)-[:SIGNED_BY]->(dept2:DEPARTMENT)
RETURN collect(DISTINCT parent.name) AS parents,
       collect(DISTINCT dept.name)     AS departments,
       collect(DISTINCT dept2.name)    AS sibling_departments,
       collect(DISTINCT co.name)       AS case_ids
"""

# Both case queries below return the *same* row shape (lists, one row per
# change order). Aggregating inside Cypher is what keeps ``LIMIT`` meaningful:
# without the ``collect`` a case would fan out into one row per department and
# silently eat the limit.
_CASE_ROW_PROJECTION = """
RETURN co.name AS case_id,
       coalesce(co.group_key, co.name) AS group_key,
       components,
       reasons,
       timepoints,
       departments
ORDER BY case_id
"""

_CYPHER_PREFETCH = (
    """
MATCH (co:CHANGE_ORDER)
OPTIONAL MATCH (co)-[:MODIFIES]->(comp:COMPONENT)
OPTIONAL MATCH (co)-[:HAS_REASON]->(reason:REASON)
OPTIONAL MATCH (co)-[:OCCURS_AT]->(tp:TIME_POINT)
OPTIONAL MATCH (co)-[:SIGNED_BY]->(dept:DEPARTMENT)
WITH co,
     collect(DISTINCT comp.name)   AS components,
     collect(DISTINCT reason.name) AS reasons,
     collect(DISTINCT tp.name)     AS timepoints,
     collect(DISTINCT dept.name)   AS departments
WHERE any(term IN $terms WHERE term <> ''
          AND (co.name CONTAINS term
               OR any(c IN components WHERE c CONTAINS term)
               OR any(r IN reasons    WHERE r CONTAINS term)
               OR any(t IN timepoints WHERE t CONTAINS term)))
"""
    + _CASE_ROW_PROJECTION
    + "LIMIT $limit\n"
)

# Used to hydrate vector hits: a Qdrant payload only carries the case id, the
# structured departments/disciplines live in the graph.
_CYPHER_CASES_BY_IDS = (
    """
UNWIND $case_ids AS cid
MATCH (co:CHANGE_ORDER {name: cid})
OPTIONAL MATCH (co)-[:MODIFIES]->(comp:COMPONENT)
OPTIONAL MATCH (co)-[:HAS_REASON]->(reason:REASON)
OPTIONAL MATCH (co)-[:OCCURS_AT]->(tp:TIME_POINT)
OPTIONAL MATCH (co)-[:SIGNED_BY]->(dept:DEPARTMENT)
WITH co,
     collect(DISTINCT comp.name)   AS components,
     collect(DISTINCT reason.name) AS reasons,
     collect(DISTINCT tp.name)     AS timepoints,
     collect(DISTINCT dept.name)   AS departments
"""
    + _CASE_ROW_PROJECTION
)


def case_row_to_meta(row: dict[str, Any]) -> EvidenceMeta:
    """One Cypher case row -> one evidence item.

    The single place where graph rows become evidence, so the plain graph
    retriever and the hybrid one cannot drift apart.
    """
    case_id = str(row.get("case_id") or "")
    departments = [d for d in (row.get("departments") or []) if d]
    components = [c for c in (row.get("components") or []) if c]
    reasons = [r for r in (row.get("reasons") or []) if r]
    timepoints = [t for t in (row.get("timepoints") or []) if t]
    return EvidenceMeta(
        evidence_id=f"CASE-{case_id}",
        entity_kinds=["CHANGE_ORDER", "COMPONENT"],
        entity_keys=[k for k in [case_id, *components] if k],
        disciplines=disciplines_for_departments(departments),
        group_keys=[row.get("group_key") or case_id],
        source="graph",
        score=1.0,
        content=(
            f"{case_id}：{'、'.join(reasons) or '（无原因记录）'}；"
            f"时间点 {'、'.join(timepoints) or '未记录'}；"
            f"变更对象 {'、'.join(components) or '未记录'}；"
            f"签收部门 {'、'.join(departments) or '未记录'}"
        ),
    )


class Neo4jRetriever:
    """Cypher-only retrieval (no embeddings on this lab machine)."""

    def __init__(self, cfg: Settings) -> None:
        self._driver = GraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password.get_secret_value()),
        )
        self._database = "neo4j"

    def close(self) -> None:
        self._driver.close()

    def ping(self) -> bool:
        """可用性探测：连接 + 认证都能过才算活着。"""
        self._run("RETURN 1 AS ok")
        return True

    # -- exception translation (the only place neo4j.* is visible) -------- #
    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """执行一条**只读**查询。

        这是读路径上**唯一**的执行入口，两道防线都在这里：

        1. `validate_read_only`：语句形态校验（fail-closed），失败抛 ``InvalidRequest``，
           调用方据此显式降级（P5）。它**放在 try 之外**——否则会被下面的异常翻译
           误报成 Neo4j 故障；
        2. ``READ_ACCESS`` 会话：服务端保证。规则是黑名单（可能漏），只读事务是白名单。

        图在摄取完成后是只读对象，写操作只属于摄取（`graph_store.py`）。
        """
        validate_read_only(query)
        try:
            with self._driver.session(
                database=self._database, default_access_mode=READ_ACCESS
            ) as session:
                return [dict(record) for record in session.run(query, **params)]
        except Exception as exc:  # neo4j.* -> our hierarchy
            name = type(exc).__name__
            if "ServiceUnavailable" in name or "SessionExpired" in name:
                raise TransientError(f"Neo4j 不可用: {exc}") from exc
            raise PermanentExternalError(f"Neo4j 查询失败: {exc}") from exc

    async def _arun(self, query: str, **params: Any) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._run, query, **params)

    async def arun(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Public escape hatch for callers that need their own Cypher.

        Still the *only* place ``neo4j.*`` is visible, so composing retrievers
        can run extra queries without touching the driver themselves.
        """
        return await self._arun(query, **params)

    # -- structured case access (shared with the hybrid retriever) --------- #
    async def cases_by_terms(self, terms: list[str], limit: int) -> list[dict[str, Any]]:
        terms = [t.strip() for t in terms if t and t.strip()]
        if not terms:
            return []
        return await self._arun(_CYPHER_PREFETCH, terms=terms, limit=limit)

    async def cases_by_ids(self, case_ids: list[str]) -> list[dict[str, Any]]:
        case_ids = sorted({c for c in case_ids if c})
        if not case_ids:
            return []
        return await self._arun(_CYPHER_CASES_BY_IDS, case_ids=case_ids)

    # -- port -------------------------------------------------------------- #
    async def link(self, request: str) -> list[EntityRef]:
        rows = await self._arun(_CYPHER_LINK, request=request, limit=20)
        names = [row["name"] for row in rows if row.get("name")]
        # CONTAINS 命中后还要过一次前缀判据，否则 FR36 会把 FR3 也带出来。
        return [
            EntityRef(name=name, kind="COMPONENT", in_graph=True, graph_key=name)
            for name in names
            if not is_partial_identifier(name, request)
        ]

    async def expand(self, entities: list[EntityRef]) -> GraphExpansion:
        names = [e.name for e in entities]
        if not names:
            return GraphExpansion()
        rows = await self._arun(_CYPHER_EXPAND, names=names)
        parents: list[str] = []
        departments: list[str] = []
        cases: list[str] = []
        for row in rows:
            parents += [p for p in (row.get("parents") or []) if p]
            departments += [d for d in (row.get("departments") or []) if d]
            departments += [d for d in (row.get("sibling_departments") or []) if d]
            cases += [c for c in (row.get("case_ids") or []) if c]
        return GraphExpansion(
            parent_components=sorted(set(parents)),
            historical_departments=sorted(set(departments)),
            historical_disciplines=disciplines_for_departments(departments),
            similar_case_ids=sorted(set(cases)),
        )

    async def prefetch(self, queries: list[str], top_k: int) -> EvidenceBundle:
        rows = await self.cases_by_terms(queries, top_k)
        items = [case_row_to_meta(row) for row in rows]
        return EvidenceBundle(
            round=0,
            baseline_ids=[item.evidence_id for item in items],
            items=items,
            warnings=[] if items else ["low_evidence"],
        )
