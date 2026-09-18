"""闸门脚本自身的回归测试。

``scripts/`` 下的闸门脚本被 git 钩子与 CI 以**退出码**为唯一信号调用，
所以「结论与调用环境无关」必须是可测试的性质，而不是约定。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# 造「带泄漏的样本仓库」用的暂存目录。它**必须**落在 .gitignore 覆盖的路径下
# （见 .gitignore 的 .cache/）：样本里有一条故意像密钥的字符串，若它出现在未被
# 忽略的路径上，下一次 check_secrets.py 会把它当成真泄漏，`git add -A` 甚至
# 可能把它提交上去。
SCRATCH = ROOT / ".cache" / "test-scripts-scratch"


def _run_script(
    name: str, *args: str, io_encoding: str | None = None
) -> subprocess.CompletedProcess[bytes]:
    """以 ``python scripts/<name>`` 的方式运行闸门脚本。

    ``capture_output=True`` 让子进程的 stdout 变成**管道** —— 这正是 git 钩子、
    CI 日志采集与 ``| tee`` 的形态，也是 Python 在 Windows 上退回 locale 编码
    （本机 cp936/GBK）的条件。``io_encoding`` 把这个条件钉死，使复现不依赖
    跑测试的机器恰好是什么 locale。
    """
    env = dict(os.environ)
    if io_encoding is not None:
        env["PYTHONIOENCODING"] = io_encoding
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=False,
    )


@pytest.fixture()
def scratch_repo() -> Iterator[Path]:
    """一个空的、被 git 忽略的目录，充当 ``check_secrets.py --root``。

    刻意不用 ``tmp_path``：本项目的开发环境把工作区之外的文件写权限收得很紧，
    ``%TEMP%`` 下能建目录却写不进文件（``PermissionError``），于是 ``tmp_path``
    既有写不进的问题，pytest 自己的清理也会踩到同样的权限边界。
    """
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    try:
        yield SCRATCH
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def test_secret_scan_reports_clean_even_when_stdout_cannot_encode_the_mark() -> None:
    """管道 + GBK 时，扫描干净必须仍然 exit 0，不能因打印 ``✔`` 而失败。

    退出码是钩子唯一读得到的信号，读错一次就会让**每一次提交**都被拦下，
    并把原因误报成「疑似密钥泄漏」。
    """
    result = _run_script("check_secrets.py", io_encoding="gbk")

    stderr = result.stderr.decode("utf-8", errors="replace")
    assert b"UnicodeEncodeError" not in result.stderr, stderr
    assert result.returncode == 0, stderr
    # 结论得真的打印出来，否则钩子里只看得到「提交被阻止」，看不到为什么。
    assert "未发现密钥泄漏" in result.stdout.decode("utf-8")


def test_secret_scan_still_fails_on_a_real_leak(scratch_repo: Path) -> None:
    """闸门不能是「永远绿」：真泄漏必须 exit 1。

    这是全项目最要紧的脚本，必须有用例验证它**真的抓得到东西**，
    而不只是「跑起来没崩」。
    """
    # 刻意写成拼接而非整条字面量：否则本文件自身会被判据 2 命中，
    # 让上面那条「扫描干净」的用例红在这个 fixture 上。
    (scratch_repo / "leak.py").write_text(
        'TOKEN = "' + "sk-" + "a1b2c3d4e5f6g7h8i9j0" + '"\n', encoding="utf-8"
    )

    result = _run_script(
        "check_secrets.py", "--all", "--root", str(scratch_repo), io_encoding="gbk"
    )

    stdout = result.stdout.decode("utf-8", errors="replace")
    assert result.returncode == 1, stdout
    assert "疑似密钥泄漏" in stdout


def test_a_shebang_implies_the_executable_bit() -> None:
    """``ruff`` 的 EXE001 只在 Linux 触发，这里给它做一个本地代理。

    Windows 看不到 POSIX 可执行位，于是「有 shebang、但 git 索引里是 100644」的文件
    在本地永远绿、只在 CI 红 —— AGENTS.md §1.3 专门警告过这个坑。这里改从 git 索引读
    模式，让这个陷阱在本地就可见。

    局限：只看**已跟踪**的文件。刚建好还没 `git add` 的脚本不在索引里，覆盖不到 ——
    那属于 `git status` 该看的东西。
    """
    listing = subprocess.run(
        ["git", "ls-files", "-s", "scripts"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    offenders: list[str] = []
    for line in listing.splitlines():
        meta, _, path = line.partition("\t")
        mode = meta.split(maxsplit=1)[0]
        if mode == "100644" and (ROOT / path).read_text(encoding="utf-8").startswith("#!"):
            offenders.append(path)

    assert not offenders, (
        f"这些文件有 shebang 但 git 里不是可执行位（EXE001 只在 CI 暴露）：{offenders}"
    )
