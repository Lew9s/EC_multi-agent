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

> ⚠ **集成用例不得共享可变资源**。并发的两个 `pytest` 进程会在外部服务上互相删除对方
> 正在使用的对象：固定集合名 + `recreate=True` → `404 Collection ... doesn't exist`；
> 固定单号前缀 + teardown `purge` → 对方断言时数到 0 条。这类失败**每次失败的用例集合
> 都不同**（谁先谁后不确定），极难排查。凡是要落到外部服务或磁盘上的测试资源
> （集合名、单号前缀、临时文件），一律**按进程隔离**（进程号或 uuid 后缀），
> teardown 只回收自己那一份。宁可多发一份数据，也不要给共享资源排队加锁。

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
7. **提交进仓库的文件里只写「代码功能解释」，不写「修复性注释」**。
   *注释回答的是「这段代码在做什么、为什么必须这么做」，不是「这次改动修了什么」。*
   变更经过属于 commit message、PR 描述与 `docs/design.md` 决策记录，不属于代码。
   具体地：
   * **不要**写「此前 / 之前 / 曾经 / 原来是这样」「修掉 / 修复 / 回归 / 上一版 / 首版」
     「某次真实 run 暴露了…」「正好说反」这类复盘与对比叙述；
   * **不要**写状态值、字段、函数「原来是几个、现在收成几个」的沿革；
   * **不要**写「为什么删掉了某个探测 / 分支 / 字段」的考古式说明；
   * **不要**把 run_id、报错原文、排障命令输出、字符偏移等现场证据抄进注释；
   * **要**保留：函数/类/模块做什么、参数与返回语义、**当前**设计的不变量与理由、
     以及解释「当前设计为何如此」的决策编号（`D-xx` / `§x.y`）；
   * 同一纪律适用于测试：用例的 docstring 说明它守哪条**当前**不变量，
     不复述「优化前报了错」这类历史。
   这条约束是**不可协商**的：它保证代码只描述现在，历史只有一个来源（版本库）。

---

## 8. README 维护要求

README 是仓库的门面：读它的人应当能独立把项目跑起来，且**不得因为写法而泄漏
任何不该公开的信息**。改 README 属于 `docs` 类提交，同样走分支 + PR。

### 8.1 必须写清楚的四件事

1. **启动方式**：环境如何创建、依赖如何安装、需要哪些外部服务与密钥**变量名**、
   离线与真实两条链路各自的完整命令。命令必须**可直接复制执行**，
   不用「配置好环境后运行」这类空话；把已知的启动故障（镜像不通、端口被占用、
   包被 env 之外的 `site-packages` 遮蔽）一并写进「常见问题」。
2. **依赖**：运行时依赖、开发依赖、外部服务与模型**逐项列出**（名称 + 版本约束
   + 它在系统里干什么），并与 `pyproject.toml` / `docker-compose.yml` 保持一致。
   依赖有任何增减，同一个 PR 里同步此节。
3. **已实现的功能**：分条目列出，每条一句话说清「做了什么」，并带上依据的决策编号。
4. **待实现的功能**：同样分条目列出，并区分三种状态 —— **未实现**（如 kernel）、
   **已落地但未接线**（如元智能体的 LLM 决策层）、**待标定**（阈值）。
   功能一旦落地或接线，必须在同一个 PR 里从这一节移到上一节。

### 8.2 禁止写进 README 的内容

1. **未上传的文件名**，尤其是数据文件。本仓库是 public，业务语料不进版本库，
   因此 README **不得出现语料文件名**（`data/README.md`、`docs/` 同理）：
   需要指路时写配置项名（`DATA_DIR` / `CORPUS_FILE`），让读者去看 `.env.example`。
   本机绝对路径、内网地址、个人目录名同样不写。
2. **任何 API key 信息**：不写 key 的值、片段、前缀或格式示例，也不写申请入口链接。
   README 只允许出现**环境变量名**——读者靠它知道要配什么，真值只存在于 `.env`
   （密钥纪律见 §1.1，与本节互为补充）。
3. **与开发过程、工具使用相关的内容**：不记录「这一版改了什么」、排障过程、
   agent 工具的使用方式或对话痕迹。变更经过属于 commit message、PR 描述与
   `docs/design.md` 决策记录。
4. **把「未验证」写成「已完成」**：§1.3 与 §5 的诚实要求同样适用于 README。

### 8.3 仓库只保留项目内容

仓库里只应有：源码、测试、文档、构建与 CI 配置，以及 `AGENTS.md` 这类协作规则。
**不得提交**开发过程产物 —— 运行日志与事件日志、各类缓存、pytest 临时目录、
PR / 提交信息草稿、agent 会话记录与工具中间产物、编辑器与工具的个人配置。
`.gitignore` 已覆盖常见路径（`logs/`、`.cache/`、`.pytest_run/` 等）；
新增一类产物时要把它加进 `.gitignore`，而不是靠人工记得别 `git add`。

> 判断标准：**删掉这个文件，项目还能不能完整地构建、运行、被理解？**
> 不能删的才是项目内容。
