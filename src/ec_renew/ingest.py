"""摄取入口：``python -m ec_renew.ingest``。

实现放在 ``rag_llama/ingest.py``（与管道其余部分同处一个包）。
这里只是一层转发，让命令短到可以记住 —— 文档、`.env.example` 与 README
里写的都是这个入口。
"""

from __future__ import annotations

import sys

from .rag_llama.ingest import (  # noqa: F401  (re-export 供脚本调用)
    IngestReport,
    _build_parser,
    ingest,
    ingest_sync,
    main,
)

__all__ = ["IngestReport", "ingest", "ingest_sync", "main"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
