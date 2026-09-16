# AGENTS.md — 协作与 PR 规则

> 本文件规定**在本仓库工作的人与 AI agent 都必须遵守的流程与边界**。
> 目标：任何一次推送都可复现、可评审、可回溯，且**永不泄漏密钥**。
>
> 仓库：<https://github.com/Lew9s/EC_multi-agent>（**public**，任何提交立即可被全网检索）
>
> 本文件原名按惯例取 `AGENTS.md`（生态通用名，且与本项目参考实现 `CDIACR` 一致）。
> 若你的工具链只认 `agent.md`，自行改名即可，内容不变。

---

## 0. 先读这些

| 文档 | 作用 |
| --- | --- |
| [`docs/design.md`](docs/design.md) | 架构原则、决策记录（D-01…D-69）、待决事项（Q-01…Q-19） |
| [`docs/rag.md`](docs/rag.md) | RAG 管道：摄取、schema、检索、降级、阈值标定 |
| [`README.md`](README.md) | 快速开始与代码结构 |

**冲突时的优先级**：`docs/design.md` 的决策记录 > 本文件 > 代码注释。
若本文件与决策记录冲突，改本文件，不要改代码绕过决策。

---

## 1. 铁律（不可协商，违反即回滚）

### 1.1 密钥

1. **密钥只能存在于 `.env`**，它是唯一真源，且已被 `.gitignore` 忽略。
2. **代码里不得出现任何明文密钥**。读取只允许发生在 `src/ec_renew/config.py`，
   且必须持有为 pydantic `SecretStr`。
3. **密钥不得进入日志**：事件日志、`repr()`、`model_dump()`、异常消息、
   span 属性一律不写密钥、连接串、prompt 原文。
4. **新增密钥必须同时更新 `.env.example`**（占位值，不是真值）。
5. `scripts/check_secrets.py` 必须通过。**任何时候都不许用
   `--no-verify` / `--no-gpg-sign` 绕过它**。
   确认是误报时，在扫描器的 `ALLOWLIST` 里加一条**带理由的、只豁免单条规则**的
   例外，并让这条例外出现在 PR diff 里接受评审。
6. **密钥一旦被推送，即视为已泄漏**：立刻到服务商后台吊销并轮换，
   而不是 `git commit --amend` 假装没发生（对象已进入远端，克隆与缓存都留痕）。

### 1.2 架构不变量

这些不是风格偏好，是 `docs/design.md` 里定下的、有测试守护的约束：

| 不变量 | 含义 |
| --- | --- |
| 单向依赖 | `interface → workflow → agents/rag → ports → contracts`；`contracts` 不 import 任何业务模块 |
| Cypher 唯一出处 | 只允许出现在 `rag/graph.py` 与 `rag/graph_store.py` |
| 异常只翻译一次 | 上层不得出现 `httpx.*` / `neo4j.*` / `qdrant_client.*` 异常类型 |
| `InvariantViolation` 永不捕获 | 它代表引擎写错，重试无意义 |
| 降级必须显式 | 禁止静默回退；降级要带原因、打印、写事件日志 |
| 事实必须可回溯 | 任何事实性结论带 `evidence_ids`，且 ⊆ 本轮 registry |
| 投影是确定性函数 | `MemoryService.project()` 不得变成 LLM tool |
| 可复现 | 平行分支结果按**排序**归并；禁 LLM 摘要压缩证据 |
| 密钥不进业务层 | 业务模块只见 `Settings`，拿不到明文 |

### 1.3 质量闸门

推送前**三条必须全绿**：

```bash
python -m ruff check src tests scripts   # 静态检查
python -m pytest tests -q                # 测试（无 docker 时集成用例自动 skip）
python scripts/check_secrets.py          # 密钥扫描
```

缺依赖时：

```bash
python -m pip install -i https://pypi.org/simple -e ".[dev]"
```

> ⚠ 本机默认的 pip 镜像（清华）不通，**必须显式 `-i https://pypi.org/simple`**。

> ⚠ **Windows 上 ruff 看不到 POSIX 可执行位**：`EXE001`（有 shebang 但文件不可执行）
> 只在 Linux / CI 暴露，本地会一直是绿的。给脚本加可执行位必须用
> `git update-index --chmod=+x <path>` —— 在资源管理器里改属性不会被 git 记录。
> 同理，`pip < 25.1` 会静默忽略 PEP 735 的 `[dependency-groups]`，
> 所以 dev 依赖只写成 `[project.optional-dependencies]`（extra）。
> **本地绿不等于 CI 绿**；改动构建/CI 时以 CI 结果为准。

---

## 2. 分支与提交

### 2.1 分支模型

```
main                    受保护，只接受通过 PR 的合并，禁止直接 push
└── <type>/<kebab-scope>  工作分支，例：feat/evidence-request、fix/fr3-substring
```

* 分支名 = `<type>` + `/` + 简短英文 scope，`type` 取值同下表的 commit 类型。
* 一个分支只做一件事。混装无关改动会让 PR 无法评审，也会让回滚失去粒度。
* 工作分支合并后删除。

### 2.2 提交信息（Conventional Commits）

```
<type>(<scope>): <中文或英文的祈使句摘要>

<可选正文：为什么这么改，而不是改了什么 —— 改了什么 diff 里看得见>

<可选脚注：Refs: D-63 / Closes #12 / BREAKING CHANGE: ...>
```

| type | 用于 |
| --- | --- |
| `feat` | 新能力 |
| `fix` | 缺陷修复 |
| `refactor` | 不改变外部行为的重构 |
| `docs` | 文档（含 `docs/design.md` 决策记录） |
| `test` | 测试 |
| `build` | 依赖、打包、CI、docker |
| `chore` | 杂项（格式化、忽略规则） |

`scope` 用模块名，例：`rag`、`agents`、`workflow`、`interface`、`config`、`ci`、`deps`。

* **摘要不超过 72 字符**，不以句号结尾。
* 正文写**理由**（为什么），不重复 diff（是什么）。
* 一次提交一个逻辑变更。`fix` 类提交应在正文里写出**可复现的失败现象**。

示例：

```
fix(rag): FR36 不再把图里的 FR3 也 link 出来

CONTAINS 匹配下，请求里的 FR36 会命中图中的 FR3，导致 expand() 推出
错误的专业集。新增 is_partial_identifier 判据：以数字结尾的名字，
若在请求里每次出现后面都还跟着数字，则视为部分匹配并丢弃。

Refs: D-64
```

---

## 3. PR 规则

### 3.1 标题

与 squash 后的提交信息同格式：`<type>(<scope>): <摘要>`。
仓库默认使用 **squash merge**，PR 标题就是最终的 main 提交信息。

### 3.2 描述（必须填，模板见 `.github/pull_request_template.md`）

PR 描述必须回答四件事：

1. **为什么做这个改动** —— 关联的决策编号（`D-xx`）或待决事项（`Q-xx`）。
   若引入了新的架构决策，**先在 `docs/design.md` §10 增补 `D-xx`**，再提 PR。
2. **改了什么** —— 按模块列出，标出对外契约（`contracts.py` / `ports.py`）的变化。
3. **怎么验证的** —— 具体命令与**实际输出**，不是"应该没问题"。
   涉及真实模型/存储时，贴关键数字（点数、图计数、相似度、共识分）。
4. **已知限制 / 未验证项** —— 有意留白的部分要写出来，不要等人问。

> 本项目明确区分「已验证」与「未验证」。填 `docs/rag.md` §9 的表格时保持诚实：
> 没跑过真机的路径就写「未验证」，不要因为代码写完就标成完成。

### 3.3 检查单（PR 模板内含，逐条勾选）

* [ ] `ruff` / `pytest` / `check_secrets` 三条闸门本地全绿
* [ ] 未提交 `.env`、事件日志、缓存或其它运行产物
* [ ] 新增配置项**同时**更新了 `config.py` 与 `.env.example`
* [ ] 涉及设计取舍时，`docs/design.md` 决策记录已增补
* [ ] 新增/修改的行为有对应测试；修 bug 的 PR 含**先失败后通过**的用例
* [ ] 对外契约变化已在 PR 描述中显式标注
* [ ] 降级路径是显式的（带原因 + 事件日志），没有静默回退

### 3.4 评审

* **禁止自我合并**（仓库 owner 也适用）：至少一次他人 approve。
  个人项目无法找到评审人时，**在 PR 里留下自查记录**：
  贴出三条闸门的真实输出，并说明为什么该改动风险可控。
* 评审关注点按优先级：**密钥与安全** > **架构不变量** > **正确性** > **可复现性** > 风格。
* 评审意见用 `suggestion` 提具体改法，避免"这里不太好"这类无法执行的评论。
* 作者负责解决评论后再请求复审；未解决的评论不得合并。

### 3.5 变更类型 → 必须同步更新的文件

提交前对照此表，**漏更新即为不完整变更**：

| 改动 | 必须同步更新 |
| --- | --- |
| 新增/修改环境变量 | `config.py` 的字段与 `from_env`、`.env.example`、（必要时）`.env` |
| 新增依赖 | `pyproject.toml`，并在 PR 里说明**实测过的版本组合** |
| 新增检索后端 / 改变降级行为 | `rag/factory.py`、`docs/rag.md` §7、README |
| 改 Neo4j schema 或 Cypher | `rag/graph.py`、`rag/graph_store.py`、`docs/rag.md` §4、`design.md` §0.3 |
| 改 Qdrant 集合 / 维度 / payload | `rag/vector_store.py`、`docs/rag.md` §5，并提醒**换维度必须 `--recreate`** |
| 改共识/投影/证据语义 | `contracts.py`、`memory.py`、**`docs/design.md` 决策记录**、`tests/test_constraints.py` |
| 改阈值/策略默认值 | `config.py`、`.env.example`，并给出**标定依据**（如 `calibrate` 输出） |
| 新增端口或适配器 | `ports.py`、`docs/design.md` §3、README 的依赖方向图 |

---

## 4. 标准推送流程

```bash
# 0) 一次性：启用仓库自带的 git 钩子（见 §6）
python scripts/install_hooks.py

# 1) 从最新的 main 开分支
git switch main && git pull --ff-only
git switch -c feat/your-change

# 2) 改代码，然后跑三条闸门
python -m ruff check src tests scripts
python -m pytest tests -q
python scripts/check_secrets.py

# 3) 确认暂存内容里没有运行产物
git status --short
git add -A
git diff --cached --stat

# 4) 提交（钩子会再跑一次扫描）
git commit -m "feat(scope): 摘要"

# 5) 推送并开 PR
git push -u origin feat/your-change
gh pr create --fill --base main        # 或到网页开 PR
```

**直接 `git push origin main` 是禁止的。** 若分支保护尚未配置，也必须自觉走 PR。

### 4.0 唯一例外：空仓库的首次提交

仓库还没有任何提交时无法开 PR（没有 base 分支可比较），因此**首次提交是唯一
允许直接落到 `main` 的操作**。它由仓库初始化时一次性完成，之后立即回归
「一律走 PR」。首次提交完成后建议手动开启分支保护：

```
Settings → Branches → Add branch protection rule
  Branch name pattern: main
  ☑ Require a pull request before merging
  ☑ Require status checks to pass (选 CI / gates)
  ☑ Require conversation resolution before merging
```

分支保护只能在 GitHub 网页或具备 admin 权限的 API 上配置 —— 本项目使用的
推送凭据通常只有 write 权限，配不了，需人工确认一次。

### 4.1 合并前

* CI（`.github/workflows/ci.yml`）三条闸门全绿。
* 检查单逐条勾选。
* 至少一次 approve（或按 §3.4 留下自查记录）。
* 用 **squash merge**，合并后删除分支。

---

## 5. 禁止事项

* 提交 `.env` 或任何真实密钥；用 `--no-verify` 绕过检查。
* 直接 push `main`；force push 已推送的公共分支。
* 提交 `logs/`、`.cache/`、`.pytest_run/`、`__pycache__/` 等运行产物。
* 在 `contracts.py` 里 import 业务模块；在 `rag/graph.py` / `rag/graph_store.py` 之外写 Cypher。
* 用裸 `except:` / `except Exception: pass` / `except: return 0` 吞掉错误。
* 捕获 `InvariantViolation` 或 `asyncio.CancelledError`。
* 让上层代码看到第三方库异常类型（`httpx.*` / `neo4j.*` / `qdrant_client.*`）。
* 静默降级：回退必须带原因、打印、写事件日志。
* 把概率性结果（LLM 摘要、`confidence`）写进共识计分或证据正文。
* 把「未验证」写成「已完成」。诚实标注限制是本项目的硬性要求。

---

## 6. 本地钩子

仓库自带 `.githooks/`，安装一次即可：

```bash
python scripts/install_hooks.py     # 等价于 git config core.hooksPath .githooks
```

`pre-commit` 钩子做两件事：

1. `scripts/check_secrets.py --staged` —— **只扫暂存区**，这是防泄漏最有效的一道闸门；
2. 对暂存的 `.py` 跑 `ruff check`（若本机装了 ruff）。

钩子只做快速检查，**完整测试仍由人工与 CI 负责**（钩子里跑全量测试会拖慢每一次提交，
最终一定会被人用 `--no-verify` 绕过，反而更糟）。

---

## 7. 给 AI agent 的额外约束

1. **先读 `docs/design.md` 再改代码**。该文件里的 `[已定]` 决策不得在未被要求时推翻。
2. **不要顺手重构无关文件**。既有文件的历史 lint 债不因你碰了同目录就该一起清。
3. **声称完成前必须真的跑过**。说明「怎么验证的」时贴真实输出；
   无法验证的（缺 key、缺服务）必须显式写成未验证项。
4. **发现既有实现有缺陷时先报告再改**，并在 PR 描述里写清现象与证据。
5. **不要为了通过检查而放宽检查**。闸门失败时先判断是真问题还是规则过严：
   规则过严就在 PR 里讨论并留下理由，不要就地注释掉。
6. **不要创建不必要的分支或提交**。一次逻辑变更一个提交，不要把格式化、
   重命名与功能改动混在同一个提交里。
