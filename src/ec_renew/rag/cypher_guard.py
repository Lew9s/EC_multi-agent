"""Cypher 的**只读**校验（fail-closed）。

读路径（`Neo4jRetriever`）只应执行只读查询。校验分两道，因为它们失效模式互补：

* **本模块（黑名单）**：语句形态层面的规则校验，快、失败明确，但黑名单永远可能漏；
* **数据库只读事务（白名单）**：`graph.py` 以 ``READ_ACCESS`` 开会话，由服务端拒绝任何写。
  它是权威兜底——本模块判过不等于服务端放行。

因此本模块的取舍是**宁可误拒**：写关键字出现在字符串或注释里也会被拒。规则校验的用途是
「把明显不对的语句挡在执行之前并给出可读原因」，不是「精确解析 Cypher」。

`CALL` 只放行一小段只读过程白名单；其余一律拒绝（包括 `apoc.*` 与写子查询）。
"""

from __future__ import annotations

import re

from ..errors import InvalidRequest

#: 出现即拒的**子句级**写关键字。按词边界匹配，且只列「能出现在语句中段」的写子句：
#: 像 `CREATE INDEX` / `DROP CONSTRAINT` / `STOP DATABASE` / `GRANT …` 这类，首词就已经被
#: `READ_STARTERS` 挡住；`USING INDEX` 是只读提示，`db.index.*` 是只读过程名——都不该误伤。
FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "FOREACH",
    "LOAD",
    "DROP",
)

#: 语句必须以其中之一开头（读子句）。`CALL` 另受过程白名单约束。
READ_STARTERS: frozenset[str] = frozenset(
    {"MATCH", "OPTIONAL", "UNWIND", "WITH", "RETURN", "CALL"}
)

#: 允许 `CALL` 的只读过程前缀。白名单而非黑名单：写过程如 `apoc.create.node` 不在其中。
READ_PROCEDURES: tuple[str, ...] = (
    "db.labels",
    "db.relationshipTypes",
    "db.propertyKeys",
    "db.schema",
    "db.index.fulltext.queryNodes",
    "db.index.fulltext.queryRelationships",
)

_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_WORD_RE = re.compile(r"[A-Za-z_]+")


def _strip_comments(query: str) -> str:
    """去掉注释再检查**内容**；注释本身不构成放行理由，见模块 docstring。"""
    return _COMMENT_RE.sub(" ", query)


def _statements(query: str) -> list[str]:
    parts = [part.strip() for part in query.split(";")]
    return [part for part in parts if part]


def validate_read_only(query: str) -> str:
    """校验一条 Cypher 是否只读。通过则原样返回，否则抛 ``InvalidRequest``。

    抛 ``InvalidRequest`` 而不是普通 ``ValueError``：调用方据此**显式降级**（P5），
    绝不能静默执行原语句，也不该把它当引擎错误重试。
    """
    text = _strip_comments(query).strip()
    if not text:
        raise InvalidRequest("Cypher 为空")

    statements = _statements(text)
    if len(statements) != 1:
        # 多语句是「先读后写」拼接的典型载体，直接拒。
        raise InvalidRequest(f"Cypher 只允许单条语句，收到 {len(statements)} 条")

    statement = statements[0]
    first = _WORD_RE.match(statement)
    if first is None or first.group(0).upper() not in READ_STARTERS:
        raise InvalidRequest("Cypher 必须以只读子句开头（MATCH / OPTIONAL / UNWIND / WITH / RETURN / CALL）")

    upper = statement.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            raise InvalidRequest(f"Cypher 含写或管理关键字 {keyword}，只读路径拒绝执行")

    for match in re.finditer(r"\bCALL\s+([A-Za-z_][\w.]*)", statement, re.IGNORECASE):
        procedure = match.group(1)
        if not procedure.startswith(READ_PROCEDURES):
            raise InvalidRequest(f"Cypher 调用了非白名单过程 {procedure}")

    return query


def is_read_only(query: str) -> bool:
    """``validate_read_only`` 的布尔形式，便于在断言与测试里表达。"""
    try:
        validate_read_only(query)
    except InvalidRequest:
        return False
    return True
