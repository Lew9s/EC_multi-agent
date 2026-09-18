"""`cypher_guard`：读路径只执行只读 Cypher（fail-closed）。

两道防线各测一半，另一半在集成用例里（`tests/test_rag.py` 用真机验证
``READ_ACCESS`` 会话确实拒绝写）：

1. **规则**（本文件）：写关键字、多语句、非白名单过程一律拒；
2. **服务端**（`test_rag.py::test_a_write_query_is_refused_by_the_read_only_session`）。

另外断言**现有常量全部通过**——这条防的是「未来有人往读路径里加一条写查询」：
`_run` 是唯一执行入口且必经守卫，所以常量不合规会在第一次调用时炸，但那时已经晚了
（可能在真机上才发现）。这里提前把它变成单元测试。
"""

from __future__ import annotations

import pytest

from ec_renew.errors import InvalidRequest
from ec_renew.rag import graph
from ec_renew.rag.cypher_guard import is_read_only, validate_read_only

REJECTED: tuple[tuple[str, str], ...] = (
    ("写节点", "CREATE (n:CHANGE_ORDER {name: 'X'}) RETURN n"),
    ("MERGE 后删", "MATCH (n:CHANGE_ORDER {name: 'X'}) DETACH DELETE n"),
    ("改属性", "MATCH (n:CHANGE_ORDER) SET n.name = 'Y' RETURN n"),
    ("删属性", "MATCH (n:CHANGE_ORDER) REMOVE n.name RETURN n"),
    ("写子查询", "CALL { CREATE (n:Probe) } RETURN 1 AS ok"),
    ("写过程", "CALL apoc.create.node(['Probe'], {}) YIELD node RETURN node"),
    ("管理语句", "DROP INDEX ON :CHANGE_ORDER(name)"),
    ("建约束", "CREATE CONSTRAINT c FOR (n:CHANGE_ORDER) REQUIRE n.name IS UNIQUE"),
    ("批量写入", "UNWIND [{name:'A'}] AS row MERGE (n:CHANGE_ORDER {name: row.name})"),
    ("先读后写拼接", "MATCH (n:CHANGE_ORDER) RETURN n LIMIT 1; MATCH (n) DETACH DELETE n"),
    ("FOREACH 内写", "MATCH (n:CHANGE_ORDER) FOREACH (x IN [1] | SET n.flag = x) RETURN n"),
    ("LOAD CSV", "LOAD CSV FROM 'file:///x.csv' AS row RETURN row"),
    ("空语句", "   "),
    ("非读子句开头", "EXPLAIN MATCH (n) RETURN n"),
)

ALLOWED: tuple[str, ...] = (
    "MATCH (n:CHANGE_ORDER) RETURN n.name LIMIT 10",
    "OPTIONAL MATCH (n)-[:MODIFIES]->(c:COMPONENT) RETURN c.name",
    "UNWIND $names AS name MATCH (c:COMPONENT {name: name}) RETURN c.name",
    "CALL db.labels() YIELD label RETURN label",
    "CALL db.index.fulltext.queryNodes('idx', $q) YIELD node RETURN node",
    "MATCH (n) RETURN n LIMIT 1;",
    "RETURN 1 AS ok",
    "// 注释里出现 SETTING 不算关键字\nMATCH (n) RETURN n LIMIT 1",
)


@pytest.mark.parametrize(("label", "query"), REJECTED, ids=[item[0] for item in REJECTED])
def test_a_write_capable_statement_is_rejected(label: str, query: str) -> None:
    assert not is_read_only(query), f"{label} 必须被拒"
    with pytest.raises(InvalidRequest):
        validate_read_only(query)


@pytest.mark.parametrize("query", ALLOWED)
def test_a_read_only_statement_is_accepted(query: str) -> None:
    assert validate_read_only(query) == query, "通过校验的语句应原样返回"


def test_every_read_path_constant_passes_the_guard() -> None:
    """读路径里的 Cypher 全部是常量，逐条必须是只读的。

    这条是「未来改动」的护栏：`_run` 是唯一执行入口且必经守卫，常量若含写语句，
    第一次真机调用才会炸——那时已经晚了。这里提前拦住。
    """
    constants = {name: value for name, value in vars(graph).items() if name.startswith("_CYPHER")}
    assert constants, "没找到读路径常量，用例失去意义"
    for name, query in sorted(constants.items()):
        assert is_read_only(query), f"{name} 含非只读语句"


def test_the_accepted_form_is_returned_verbatim() -> None:
    """校验器不重写语句：它只裁决，不改写——改写会让「执行了什么」与审计记录不一致。"""
    query = "MATCH (n:CHANGE_ORDER) RETURN n.name LIMIT 3"
    assert validate_read_only(query) == query


# --------------------------------------------------------------------------- #
# 第二道防线：服务端只读事务（规则是黑名单，可能漏；只读事务是白名单）
# --------------------------------------------------------------------------- #


def _neo4j_reachable() -> bool:
    import socket

    sock = socket.socket()
    sock.settimeout(2)
    try:
        sock.connect(("127.0.0.1", 7687))
        return True
    except OSError:
        return False
    finally:
        sock.close()


@pytest.mark.skipif(not _neo4j_reachable(), reason="需要 docker 部署的 Neo4j")
def test_a_write_is_refused_by_the_read_only_session() -> None:
    """**绕过守卫**直接对只读会话跑写语句：必须被服务端拒绝，且图里不留痕。

    这条与规则校验正交——它验证的是「即使规则漏了，也写不进去」。
    """
    from neo4j import READ_ACCESS
    from neo4j.exceptions import Neo4jError

    from ec_renew.config import settings
    from ec_renew.rag.graph import Neo4jRetriever

    retriever = Neo4jRetriever(settings)
    try:
        # 故意用到 `_driver` / `_run` 与守卫之后的层：不绕过 `_run` 就只测到规则、测不到服务端。
        with (
            retriever._driver.session(database="neo4j", default_access_mode=READ_ACCESS) as session,
            pytest.raises(Neo4jError),
        ):
            session.run("CREATE (n:__readonly_probe__) RETURN n").consume()

        rows = retriever._run("MATCH (n:__readonly_probe__) RETURN count(n) AS c")
        assert rows == [{"c": 0}], "只读会话竟写进了图"
    finally:
        retriever.close()
