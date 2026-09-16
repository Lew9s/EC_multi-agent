#!/usr/bin/env python3
"""安装仓库自带的 git 钩子。

等价于 ``git config core.hooksPath .githooks``，但多做了两件容易漏掉的事：

1. 在支持 chmod 的平台上给钩子加可执行位；
2. 若钩子已被跟踪，同步 **git 索引里的**可执行位 —— 否则 Linux/macOS 协作者
   clone 下来会得到一个「存在但不可执行」的钩子，而 git 对此**完全不报错**，
   于是他们的提交静默地不过闸门。

用法::

    python scripts/install_hooks.py
    python scripts/install_hooks.py --uninstall
"""

from __future__ import annotations

import argparse
import stat
import subprocess
import sys
from pathlib import Path

HOOKS_DIR = ".githooks"
HOOK_FILE = "pre-commit"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )


def _ensure_executable(root: Path) -> None:
    hook = root / HOOKS_DIR / HOOK_FILE
    if not hook.is_file():
        return
    try:
        hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass  # Windows 上通常无意义，git 自己会按 sh 解释钩子

    # 只有已被跟踪时才需要同步索引位。
    tracked = _git(root, "ls-files", "--error-unmatch", f"{HOOKS_DIR}/{HOOK_FILE}")
    if tracked.returncode == 0:
        _git(root, "update-index", "--chmod=+x", f"{HOOKS_DIR}/{HOOK_FILE}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="安装/卸载仓库自带的 git 钩子")
    parser.add_argument("--uninstall", action="store_true", help="恢复 git 默认钩子路径")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    if _git(root, "rev-parse", "--is-inside-work-tree").returncode != 0:
        print(f"不是 git 仓库：{root}", file=sys.stderr)
        return 2

    if args.uninstall:
        result = _git(root, "config", "--unset", "core.hooksPath")
        print("已恢复默认钩子路径" if result.returncode == 0 else "本来就没设置过")
        return 0

    if not (root / HOOKS_DIR / HOOK_FILE).is_file():
        print(f"找不到钩子文件：{HOOKS_DIR}/{HOOK_FILE}", file=sys.stderr)
        return 2

    result = _git(root, "config", "core.hooksPath", HOOKS_DIR)
    if result.returncode != 0:
        print(f"设置 core.hooksPath 失败：{result.stderr.strip()}", file=sys.stderr)
        return 2

    _ensure_executable(root)
    print(f"✔ 钩子已启用：core.hooksPath = {HOOKS_DIR}")
    print("  每次 git commit 会先跑：scripts/check_secrets.py --staged（+ ruff，若已安装）")
    print("  规则见 AGENTS.md §6；需要跳过时请先确认不是真泄漏。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
