"""技能：可被**多个 agent 复用**的能力包（目录形式，一个技能一个目录）。

一个技能目录里有什么
--------------------
| 文件 | 作用 |
| --- | --- |
| ``SKILL.md`` | 能力说明：什么时候用、输入输出契约、边界与失败方式 |
| ``__init__.py`` | 该技能的 ``SkillSpec`` 与对外导出 |
| 其它 | 技能自己的实现（``prompt.py`` / ``runner.py`` / ``personas.py`` …） |

为什么要显式注册
----------------
``_registry()`` 里一个技能一行，**不在 import 时扫描文件系统**：旧实现在 import 时读文件、
建图、调 ``nest_asyncio.apply()``，是 ``design.md`` §0.1 明确列为重写依据的问题之一。
显式注册也让「有哪些能力」可被静态检查，而不是靠目录里恰好放了什么。

技能与 agent 的关系
-------------------
技能是**方法**（怎么做评审），agent 是**执行者**（谁来做、能做什么）。§3.1 规定元/子智能体
共用一份 ``AgentRuntime``；技能则是同一执行者可以装载的不同能力，专家之间靠 persona 区分，
靠技能复用同一套「怎么做」。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillSpec:
    """技能元数据（§3.1 里 ``AgentSpec`` 五项数据中「可复用的那部分」）。"""

    name: str
    version: str
    description: str
    inputs: str  # 输入 schema 名（来自 contracts）
    outputs: str  # 输出 schema 名（来自 contracts）
    tools: tuple[str, ...] = ()


def _registry() -> dict[str, SkillSpec]:
    """显式注册表：**新增技能 = 这里加一行**（不做文件系统扫描）。"""
    from .expert_review import SKILL as expert_review

    return {expert_review.name: expert_review}


def available_skills() -> tuple[str, ...]:
    return tuple(sorted(_registry()))


def load_skill(name: str) -> SkillSpec:
    registry = _registry()
    if name not in registry:
        raise KeyError(f"未知技能：{name!r}（已注册：{sorted(registry)}）")
    return registry[name]
