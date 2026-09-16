#!/usr/bin/env python3
"""提交前密钥扫描 —— 推送流程的第一道闸门。

为什么需要它
------------
``.env`` 已被 ``.gitignore`` 忽略，但「忽略」只挡得住那一个文件名：

* ``.env`` 里的 key 被复制进某个 ``.py`` / ``.md`` / notebook；
* 新增了 ``.env.local`` / ``.env.prod`` 而没同步加进 ``.gitignore``；
* 把 key 粘进了 ``docker-compose.yml`` 或 README 的示例里。

这类事故一旦推送到公开仓库就无法真正回收，所以必须由**自动检查**兜底，
而不是靠"我记得没提交"。

两重判据
--------
1. **精确值比对**（最强）：读 ``.env`` 里每个疑似密钥的值，在待提交文件里
   逐字查找。智谱的 key 形如 ``<id>.<secret>``，不匹配任何通用正则，
   只有这一重能抓到它。
2. **通用模式**：``sk-`` 开头的密钥、``ghp_`` 开头的 token、``AKIA`` 开头的
   AWS key、私钥块（BEGIN … PRIVATE KEY）等已知格式，用来抓
   「不属于本仓库 .env」的泄漏。

掩码形态（``sk-xxxxxxxx`` / ``<your-key>`` / ``changeme``）会被识别为占位符
并跳过，否则 ``.env.example`` 会永远误报。

输出永不回显密钥本身，只给长度与前缀 —— 扫描器自己不能变成泄漏源。

用法::

    python scripts/check_secrets.py            # 扫描「会被提交的文件」
    python scripts/check_secrets.py --all      # 连已忽略的文件一起扫（更严）
    python scripts/check_secrets.py --staged   # 只扫暂存区（pre-commit 钩子用）

退出码：0 = 干净，1 = 发现问题，2 = 自身错误。
输出编码固定为 UTF-8，不随控制台 / 管道的 locale 变化（见 ``scripts/_console.py``）。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from _console import force_utf8_output

# 环境变量名里出现这些词就当成密钥，取它的值做精确比对。
SECRET_NAME_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")

# 明显是占位符的值不参与精确比对，否则 .env.example 会一直误报。
PLACEHOLDER_RE = re.compile(
    r"""^(?:
        x{4,}|X{4,}|\*{3,}|\.{3,}|-{3,}          # 掩码
      | <[^>]*>|\$\{[^}]*\}|\{\{[^}]*\}\}        # 模板占位
      | (?:change|your|my|some|placeholder|example|sample|dummy|fake|test|todo|null|none|empty)[-_]?\w*
      | changeme | change_me | none | null | true | false
    )$""",
    re.IGNORECASE | re.VERBOSE,
)

# 已知密钥格式。「分类名 -> 正则」。
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("OpenAI/DeepSeek 风格密钥", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("GitHub fine-grained PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("AWS Access Key ID", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Google API Key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("私钥块", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("硬编码 Bearer", re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}")),
)

# 这些文件名本身就不该进版本库（.env.example 例外，见 is_allowed_file）。
FORBIDDEN_NAMES = frozenset(
    {".env", ".env.local", ".env.prod", ".env.production", ".env.dev", "credentials",
     "credentials.json", "secrets.json", "id_rsa", "id_ed25519", "id_ecdsa", ".netrc"}
)
FORBIDDEN_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".ppk", ".ovpn")

ALLOWED_FILES = frozenset({".env.example", ".env.sample", ".env.template"})

# 经复核的例外：必须逐条写明理由，且**只豁免指定规则**，不做整文件放行。
# 加条目等于一次代码评审，请勿为了让检查变绿而随手添加。
ALLOWLIST: tuple[tuple[str, str, str], ...] = (
    (
        "docs/design.md",
        "泄漏 .env 的 NEO4J_PASSWORD",
        (
            "该文件在 §0.1 引用旧实现硬编码的弱口令作为重写依据，"
            "属被批判的历史缺陷记录，不是本项目的凭据（故此处不复述该字面量）。"
        ),
    ),
)

MAX_FILE_BYTES = 2_000_000  # 超过就跳过：本仓库不该有巨大文本，且避免拖慢 CI


def allowed(path: str, rule: str) -> bool:
    return any(path == p and r in rule for p, r, _ in ALLOWLIST)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    preview: str

    def render(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"  {where}  [{self.rule}]  {self.preview}"


def redact(value: str) -> str:
    """只暴露长度和极少字符，够定位、不足以还原。"""
    value = value.strip()
    if len(value) <= 8:
        return f"<{len(value)} 字符，已隐去>"
    return f"{value[:3]}…{value[-2:]}（{len(value)} 字符）"


def looks_like_placeholder(value: str) -> bool:
    """掩码/模板值不算泄漏，否则 .env.example 会永远报红。"""
    if PLACEHOLDER_RE.match(value):
        return True
    if re.fullmatch(r"[xX*.\-_]+", value):
        return True
    body = re.sub(
        r"^(?:sk-|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|AKIA|AIza|xox[baprs]-)", "", value
    )
    stripped = re.sub(r"[^A-Za-z0-9]", "", body)
    if not stripped:
        return True
    # sk-xxxx… / xxxx.yyyy 这类：去掉前缀后只剩一两种重复字符。
    return len(set(stripped)) <= 2 and len(stripped) >= 8


def load_env_secrets(env_path: Path) -> dict[str, str]:
    """从 .env 提取「名字像密钥且值不像占位符」的条目。"""
    if not env_path.is_file():
        return {}
    secrets: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip("'\"")
        if not value or not any(hint in name.upper() for hint in SECRET_NAME_HINTS):
            continue
        if PLACEHOLDER_RE.match(value):
            continue
        secrets[name] = value
    return secrets


def candidate_files(root: Path, mode: str) -> list[Path]:
    """待检查的文件集合。

    ``committed`` 用 ``git ls-files``（含未跟踪但未被忽略的），这才是
    「本次推送真正会带上去的东西」；不在仓库里时退化为手动遍历。
    """
    if mode == "staged":
        args = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    else:
        args = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    try:
        out = subprocess.run(
            args, cwd=root, capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return _walk(root)

    tracked = [root / line.strip() for line in out.splitlines() if line.strip()]
    if mode == "all":
        tracked = [p for p in _walk(root) if ".git" not in p.parts]
    return [p for p in tracked if p.is_file()]


def _walk(root: Path) -> list[Path]:
    skip_dirs = {".git", "__pycache__", ".ruff_cache", ".pytest_cache", ".venv", "node_modules"}
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and not any(part in skip_dirs for part in p.parts)
    ]


def is_allowed_file(path: Path) -> bool:
    return path.name in ALLOWED_FILES


def scan_file(path: Path, root: Path, secrets: dict[str, str]) -> list[Finding]:
    rel = path.relative_to(root).as_posix()
    findings: list[Finding] = []

    if not is_allowed_file(path) and (
        path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES
    ):
        return [Finding(rel, 0, "禁止提交的文件类型", "该文件应加入 .gitignore")]

    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [Finding(rel, 0, "读取失败", str(exc)[:80])]

    lines = text.splitlines()

    # 判据 1：本仓库 .env 里的真实值出现在任何待提交文件中。
    for name, value in secrets.items():
        if value in text:
            rule = f"泄漏 .env 的 {name}"
            if allowed(rel, rule):
                continue
            for number, line in enumerate(lines, start=1):
                if value in line:
                    findings.append(Finding(rel, number, rule, redact(value)))
                    break

    # 判据 2：通用密钥格式。
    for label, pattern in PATTERNS:
        if allowed(rel, label):
            continue
        for number, line in enumerate(lines, start=1):
            match = pattern.search(line)
            if match and not looks_like_placeholder(match.group(0)):
                findings.append(Finding(rel, number, label, redact(match.group(0))))

    return findings


def main(argv: list[str] | None = None) -> int:
    # 先钉编码：本脚本的结论可能是「干净」（退出码 0），若结论本身打印不出来
    # 就会变成 1，而 pre-commit 钩子只读退出码 —— 见 scripts/_console.py。
    force_utf8_output()

    parser = argparse.ArgumentParser(
        prog="python scripts/check_secrets.py",
        description="扫描将要提交的文件，确认没有密钥泄漏",
    )
    parser.add_argument("--root", default=".", help="仓库根目录（默认当前目录）")
    parser.add_argument("--mode", choices=["committed", "staged", "all"], default="committed")
    parser.add_argument("--staged", action="store_true", help="等价于 --mode staged")
    parser.add_argument("--all", action="store_true", help="等价于 --mode all")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    mode = "staged" if args.staged else "all" if args.all else args.mode
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"根目录不存在：{root}", file=sys.stderr)
        return 2

    secrets = load_env_secrets(root / ".env")
    files = candidate_files(root, mode)
    findings = [f for path in files for f in scan_file(path, root, secrets)]

    if findings:
        print(f"✖ 发现 {len(findings)} 处疑似密钥泄漏（扫描 {len(files)} 个文件，模式 {mode}）")
        for finding in findings:
            print(finding.render())
        print()
        print("处理方式：把密钥移出被跟踪的文件，改从 .env 读取（见 .env.example）。")
        print("若确认是误报，请在 scripts/check_secrets.py 中收紧规则，而不是 --no-verify 跳过。")
        return 1

    if not args.quiet:
        hint = "（.env 中有 " + str(len(secrets)) + " 个密钥值参与比对）" if secrets else "（未发现 .env）"
        print(f"✔ 未发现密钥泄漏：扫描 {len(files)} 个文件，模式 {mode} {hint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
