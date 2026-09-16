"""脚本输出的编码兜底。

本模块被 import，不作为脚本执行，因此**刻意不带 shebang**：带了就是 ruff 的
EXE001（有 shebang 但文件没有可执行位），而这条规则**只在 Linux / CI 暴露** ——
Windows 上 ruff 看不到 POSIX 可执行位，本地会一直是绿的（AGENTS.md §1.3 记着这个
坑；本 PR 的第一次 CI 正是踩在它上面红的）。

为什么需要它
------------
本目录下的脚本用中文和 ``✔`` / ``✖`` 报告结论，而 Python 在 Windows 上
**只对真正的控制台**走 PEP 528 的 UTF-8；stdout 一旦是管道或重定向，就退回
locale 编码（本机是 cp936/GBK）。于是同一个脚本：

* 在终端里跑 → 一切正常；
* 被 git 钩子、CI 日志采集、``| tee`` 调用 → ``print("✔ …")`` 抛
  ``UnicodeEncodeError``，**退出码变成 1**。

对 ``check_secrets.py`` 尤其致命：扫描结果本来是干净的，闸门却报红；而
``.githooks/pre-commit`` 只读退出码，于是**每一次提交都会被拦下**，且报错
信息一律指向「疑似密钥泄漏」——与真实原因（控制台编码）完全无关。

闸门的结论必须与调用环境长什么样无关，所以这里显式把编码钉成 UTF-8。
这与 CI（Linux，本来就是 UTF-8）行为一致；Windows 控制台下 PEP 528 早已是
UTF-8，重复设置是幂等的。

用法::

    from _console import force_utf8_output

本模块假定调用方式是 ``python scripts/<脚本>.py``：此时 ``scripts/`` 位于
``sys.path[0]``，直接 import 即可，不需要包结构，也不需要手工注入 sys.path。
"""

from __future__ import annotations

import sys

__all__ = ["force_utf8_output"]


def force_utf8_output() -> None:
    """把 stdout / stderr 的编码钉成 UTF-8，且永不因编码而崩溃。

    幂等。对已被替换成不可重配对象的流（例如某些测试框架的捕获器）静默跳过：
    本函数只负责「别因为控制台编码而崩」，不负责保证输出一定是 UTF-8——
    当调用方自己接管了 stdout 时，那个决定权不该被抢过来。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # io.UnsupportedOperation 同时继承 ValueError 与 OSError；
            # 不可重配的流保持原样，绝不为了编码问题把闸门变成异常退出。
            pass
