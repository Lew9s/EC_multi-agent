"""内环：子智能体（专家）与元智能体运行时（design.md §3 模块 7）。

| 文件 | 职责 |
| --- | --- |
| ``experts.py`` | 六类专家（L0）+ 规则表选专家 + prompt 构造 + 契约重试 |
| ``memory.py`` | ``EvidenceRegistry``（run 级唯一事实源）+ 确定性 ``MemoryService``（Read / Filter / Project） |

``memory.py`` 归到本包的理由是 §5.4.10：Memory Controller 由「循环部分 + 决策部分 +
``MemoryService``」组成，而元智能体是**唯一**被授予全局记忆读权限的 agent（§8.4.4）。
``EvidenceRegistry`` 是 run 级存储（§8.4.3），与投影函数同处一文件，故随它一起归位。

依赖方向：``agents`` 依赖 ``ports`` / ``contracts``（以及横切的 ``errors``），
**不 import** ``workflow`` / ``rag`` / ``interface``（单向依赖，见 AGENTS.md §1.2）。
"""

from __future__ import annotations
