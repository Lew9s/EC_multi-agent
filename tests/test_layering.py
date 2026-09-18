"""§3.1 单向依赖与「Cypher 唯一出处」的守护。

这两条是 AGENTS.md §1.2 列的架构不变量，这里把它们变成可机检的性质：
目录名就是层名，跨层 import 就是违规。

这类违规值得单独守：`rag` 反向 import `workflow` 时，`ruff` 与功能测试都不会红——
功能照样跑，只是依赖方向悄悄烂掉。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ec_renew"

#: 层号：**数值大的可以 import 数值小的**，反向即违规（design.md §3.1）。
#:
#: 新增包时必须在这里给它定层，否则 `test_no_module_imports_upward` 会明确报错，
#: 而不是把新模块悄悄跳过。
LAYERS: dict[str, int] = {
    "ec_renew.contracts": 0,
    "ec_renew.config": 0,
    "ec_renew.errors": 0,
    "ec_renew.ports": 1,
    "ec_renew.llm": 2,
    "ec_renew.observability": 2,
    "ec_renew.session": 2,
    "ec_renew.agents": 3,
    "ec_renew.rag": 3,
    "ec_renew.workflow": 4,
    "ec_renew.interface": 5,
}

#: Cypher 只允许出现在这三个文件里（AGENTS.md §1.2），且三者都在 `rag/` 内：
#: `graph.py` 读查询、`graph_store.py` 写查询、`cypher_guard.py` **关键字守卫**——
#: 守卫必须列出关键字名，因此与查询文件同属 Cypher-aware。业务层仍不得出现 Cypher。
CYPHER_FILES = frozenset({"rag/graph.py", "rag/graph_store.py", "rag/cypher_guard.py"})
_CYPHER = re.compile(
    r"\b(?:MATCH|MERGE|UNWIND|FOREACH)\b|DETACH\s+DELETE|CREATE\s+CONSTRAINT",
    re.IGNORECASE,
)


def _modules() -> list[tuple[str, Path]]:
    """``(点分模块名, 路径)``，跳过根包 ``__init__.py``（它不属于任何一层）。"""
    out: list[tuple[str, Path]] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(SRC).parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        if not parts:
            continue  # ec_renew/__init__.py
        out.append((f"ec_renew.{'.'.join(parts)}", path))
    return out


def _layer_of(module: str) -> str | None:
    """模块归属的层：最长前缀匹配（``ec_renew.rag.corpus`` → ``ec_renew.rag``）。"""
    best: str | None = None
    for layer in LAYERS:
        matches = module == layer or module.startswith(layer + ".")
        if matches and (best is None or len(layer) > len(best)):
            best = layer
    return best


def _internal_imports(module: str, path: Path) -> list[str]:
    """该文件引用的全部 ``ec_renew`` 内部模块（含函数体内的惰性 import）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                for _ in range(node.level - 1):
                    base = base[:-1]
                prefix = ".".join(base)
                target = f"{prefix}.{node.module}" if node.module else prefix
                found.append(target)
            elif node.module:
                found.append(node.module)
        elif isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
    return [t for t in found if t == "ec_renew" or t.startswith("ec_renew.")]


def test_no_module_imports_upward() -> None:
    """层号大的可以 import 层号小的；反向即违规。"""
    violations: list[str] = []
    for module, path in _modules():
        source = _layer_of(module)
        assert source is not None, f"{module} 没有定层：新增包时请先更新 LAYERS"
        for target in _internal_imports(module, path):
            owner = _layer_of(target)
            assert owner is not None, f"{module} import 了未定层的 {target}"
            if LAYERS[owner] > LAYERS[source]:
                violations.append(
                    f"{module}（L{LAYERS[source]}）→ {target}（L{LAYERS[owner]}）"
                )
    assert not violations, "出现向上依赖：\n  " + "\n  ".join(sorted(set(violations)))


def test_contracts_depends_on_nothing() -> None:
    """``contracts`` 是唯一接口层：它不 import 任何业务模块（§3.1 硬规则）。"""
    assert _internal_imports("ec_renew.contracts", SRC / "contracts.py") == []


def test_ports_only_depends_on_contracts() -> None:
    assert set(_internal_imports("ec_renew.ports", SRC / "ports.py")) <= {"ec_renew.contracts"}


def test_cypher_stays_in_the_allowed_files() -> None:
    """扫**字符串字面量**而不是全文：注释或文档里提到 MATCH 不算写 Cypher。"""
    offenders: list[str] = []
    for module, path in _modules():
        rel = path.relative_to(SRC).as_posix()
        if rel in CYPHER_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and _CYPHER.search(node.value)
            ):
                offenders.append(f"{rel}: {node.value.strip()[:60]!r}")
    assert not offenders, "Cypher 出现在允许清单之外：\n  " + "\n  ".join(offenders)


def test_the_cypher_allowlist_is_not_vacuous() -> None:
    """反向断言：允许清单不能靠「把文件搬空」来通过——它必须真的装着 Cypher。"""
    holders = {
        path.relative_to(SRC).as_posix()
        for _module, path in _modules()
        if _CYPHER.search(path.read_text(encoding="utf-8"))
    }
    missing = sorted(CYPHER_FILES - holders)
    assert not missing, f"允许清单里没有 Cypher 的文件：{missing}"
