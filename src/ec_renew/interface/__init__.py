"""对外接口：CLI（design.md §3 模块 9）。

| 文件 | 职责 |
| --- | --- |
| ``cli.py`` | 对话式命令行：``python -m ec_renew.interface.cli`` |

本层是**装配点**：由它构造 ``LLMPort`` / ``RetrieverPort`` 的具体实现，再交给
``workflow.run``。因此只有它能同时 import ``llm``（模型适配器）与 ``rag.factory``
（检索后端选择）——业务模块只拿得到端口。

依赖方向：``interface → workflow → agents / rag → ports → contracts``。
"""

from __future__ import annotations
