"""语料解析：把变更单文本切成「一单一 Document」。

旧数据是一整个 txt，变更单之间用 `!@#$%^&*` 分隔。这里做两件事：

1. **解析**（纯函数，不依赖 llama-index）—— 产出 ``ChangeOrder``，把散在正文里的
   业务字段提成结构化元数据。解析结果是检索质量的根：``disciplines`` 决定
   ``assess_grounding`` 之后哪些专业会被激活，``case_id`` 决定证据能否回溯到图谱。
2. **转 Document**（延迟 import）—— 只有真正要喂给 LlamaIndex 时才引入依赖，
   让解析逻辑可以在没有 llama-index 的环境里单测。

幂等性：``Document.id_`` 由文件名 + 单号决定，所以重复摄取是覆盖而不是追加。
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .graph import disciplines_for_departments

# 变更单正文字段名 -> 结构化键。中英文冒号都见得到（旧数据里混用）。
_FIELD_RE = re.compile(
    r"^(单号|变更原因|变更时间点|变更内容|变更对象|签收部门)\s*[:：]\s*(.*)$"
)

# 部门之间的分隔符：旧数据用全角逗号和顿号，也见过半角逗号。
_DEPT_SPLIT_RE = re.compile(r"[，,、;；]+")

# 单号尾部的序号段：H-03-7 -> H-03（同属一个变更组）。
_GROUP_TAIL_RE = re.compile(r"-\d+$")


def group_key_for(case_id: str) -> str:
    """``H-03-7`` -> ``H-03``；``H-19`` -> ``H-19``（本来就是主单号）。

    判据是「剥掉尾部序号后**还剩不剩一个 ``-``**」：
    ``H-19`` 的 ``-19`` 是主单号的一部分，剥掉会得到毫无意义的 ``H``，
    把所有 H 开头的单子错误地并进同一组。
    """
    stripped = _GROUP_TAIL_RE.sub("", case_id)
    return stripped if "-" in stripped else case_id

DEFAULT_SEPARATOR = r"!@#\$%\^&\*"

# 固定命名空间：node id 由「文件名 + 单号」确定性映射成 UUIDv5。
# 为什么不是可读字符串：Qdrant 的 point id 只接受 uint64 或 UUID。
# 为什么用 uuid5 而不是随机 uuid4：重复摄取必须落在同一个点上（幂等）。
CORPUS_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "ec-renew/change-order-corpus")


def document_id(source_file: str, case_id: str) -> str:
    """``文件名:单号`` -> 稳定 UUID 字符串。"""
    return str(uuid.uuid5(CORPUS_NAMESPACE, f"{source_file}:{case_id}"))


@dataclass(frozen=True)
class ChangeOrder:
    """一张变更单。``text`` 是原文，永不改写（可回溯约束的前提）。"""

    case_id: str
    group_key: str
    text: str
    source_file: str
    index: int
    reason: str = ""
    time_point: str = ""
    component: str = ""
    departments: tuple[str, ...] = ()
    disciplines: tuple[str, ...] = ()
    content: str = ""
    missing: tuple[str, ...] = field(default=())

    @property
    def entity_keys(self) -> list[str]:
        return [k for k in (self.case_id, self.component) if k]

    def metadata(self) -> dict[str, Any]:
        """Qdrant payload —— 只放可索引的标量/列表，不放长正文。

        ``disciplines`` 在建库时就打好标签（design.md Q-01 选项 b），检索期零成本。
        """
        return {
            "case_id": self.case_id,
            "group_key": self.group_key,
            "source_file": self.source_file,
            "order_index": self.index,
            "reason": self.reason,
            "time_point": self.time_point,
            "component": self.component,
            "departments": list(self.departments),
            "disciplines": list(self.disciplines),
            "entity_kinds": ["CHANGE_ORDER", "COMPONENT"],
            "entity_keys": self.entity_keys,
        }


def _parse_fields(segment: str) -> dict[str, str]:
    """按行扫，字段值可以续行（变更内容常常是多行）。"""
    buckets: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in segment.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _FIELD_RE.match(line)
        if match:
            current = match.group(1)
            buckets[current] = [match.group(2).strip()]
        elif current is not None:
            buckets[current].append(line)
    return {key: " ".join(parts).strip() for key, parts in buckets.items()}


def split_segments(text: str, separator: str = DEFAULT_SEPARATOR) -> list[str]:
    """按分隔符切分，丢掉空段，顺序保持不变。"""
    return [seg.strip() for seg in re.split(separator, text) if seg.strip()]


def parse_change_orders(
    text: str,
    *,
    source_file: str = "unknown",
    separator: str = DEFAULT_SEPARATOR,
) -> list[ChangeOrder]:
    orders: list[ChangeOrder] = []
    for index, segment in enumerate(split_segments(text, separator), start=1):
        fields = _parse_fields(segment)
        case_id = fields.get("单号", "").strip() or f"AUTO-{index:03d}"
        component = fields.get("变更对象", "").strip()
        departments = tuple(
            sorted(
                {
                    part.strip()
                    for part in _DEPT_SPLIT_RE.split(fields.get("签收部门", ""))
                    if part.strip()
                }
            )
        )
        missing = tuple(
            sorted(
                name
                for name, key in (
                    ("单号", "单号"),
                    ("变更原因", "变更原因"),
                    ("变更时间点", "变更时间点"),
                    ("变更对象", "变更对象"),
                    ("签收部门", "签收部门"),
                )
                if not fields.get(key, "").strip()
            )
        )
        orders.append(
            ChangeOrder(
                case_id=case_id,
                group_key=group_key_for(case_id),
                text=segment,
                source_file=source_file,
                index=index,
                reason=fields.get("变更原因", "").strip(),
                time_point=fields.get("变更时间点", "").strip(),
                component=component,
                departments=departments,
                disciplines=tuple(disciplines_for_departments(departments)),
                content=fields.get("变更内容", "").strip(),
                missing=missing,
            )
        )
    return orders


def load_change_orders(
    path: Path | str, *, separator: str = DEFAULT_SEPARATOR
) -> list[ChangeOrder]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return parse_change_orders(text, source_file=file_path.name, separator=separator)


def to_documents(orders: Sequence[ChangeOrder]) -> list[Any]:
    """``ChangeOrder`` -> ``llama_index.core.Document``（唯一的依赖引入点）。

    node id 是 ``文件名:单号`` 的 UUIDv5：重复摄取会覆盖同一个点，
    不会在 Qdrant 里堆出重复副本。可读的单号在 ``metadata["case_id"]`` 里。
    """
    from llama_index.core import Document  # 延迟导入：解析逻辑不需要它

    documents = []
    for order in orders:
        documents.append(
            Document(
                text=order.text,
                metadata=order.metadata(),
                id_=document_id(order.source_file, order.case_id),
                excluded_embed_metadata_keys=list(order.metadata()),
                excluded_llm_metadata_keys=["entity_kinds", "entity_keys", "order_index"],
            )
        )
    return documents


def corpus_path(data_dir: Path | str, filename: str) -> Path:
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"语料不存在：{path}。请确认 DATA_DIR / CORPUS_FILE，"
            "或把语料放到 data/ 下。"
        )
    return path


def summarize(orders: Iterable[ChangeOrder]) -> dict[str, Any]:
    """摄取日志用的统计量（确定性，便于比对两次摄取是否一致）。"""
    orders = list(orders)
    by_discipline: dict[str, int] = {}
    for order in orders:
        for discipline in order.disciplines:
            by_discipline[discipline] = by_discipline.get(discipline, 0) + 1
    return {
        "orders": len(orders),
        "groups": len({o.group_key for o in orders}),
        "components": len({o.component for o in orders if o.component}),
        "incomplete": sum(1 for o in orders if o.missing),
        "by_discipline": dict(sorted(by_discipline.items())),
    }
