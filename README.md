# 工程变更方案生成系统 · 实验版 Demo

workflow（外环，确定性）+ agent（内环，有界自主）的多智能体方案生成系统，
RAG 管道用 LlamaIndex 搭建。

| 想了解 | 看这里 |
| --- | --- |
| 架构原则与决策记录（D-01…） | [`docs/design.md`](docs/design.md) |
| RAG 管道：摄取、schema、检索、降级、阈值标定 | [`docs/rag.md`](docs/rag.md) |
| 语料格式（语料本身不随仓库分发） | [`data/README.md`](data/README.md) |
| 分支、提交、PR 与质量闸门 | [`AGENTS.md`](AGENTS.md) |

---

## 1. 这个项目做什么

输入一条工程变更请求（例：某分段某污水井更换加厚板、涉及焊接，需确认合规性），系统：

1. 检索历史变更单与领域图谱，形成本轮**证据基线**（每条证据有内容寻址 ID）；
2. 六个专业视角的专家智能体**互相隔离**地评审，必要时进入第二轮匿名共识；
3. 把评审折叠成一份**逐条可回溯**的变更方案，并明确回答「交付还是交人」。

产出有两件：给人读的**方案**（Markdown），和给机器读的结构化 `RunResult`。

---

## 2. 依赖

### 2.1 运行环境

| 要求 | 说明 |
| --- | --- |
| Python | **>= 3.11**（开发与验证使用 3.12） |
| Docker | **只有真实检索链路需要**（Qdrant + Neo4j）；离线模式完全不需要 |
| 网络 | **只有真实模型链路需要**；离线模式完全不需要 |
| API Key | **只有真实模型链路需要**：一个 LLM key + 一个 embedding key |

### 2.2 Python 运行时依赖

声明见 [`pyproject.toml`](pyproject.toml)，下表是它们各自在系统里的位置：

| 包 | 版本约束 | 作用 |
| --- | --- | --- |
| `pydantic` | `>=2.10,<3` | 全部跨层 schema（`contracts.py`）；API key 以 `SecretStr` 持有 |
| `python-dotenv` | `>=1.0,<2` | 只在 `config.py` 读 `.env` |
| `httpx` | `>=0.27,<1` | 直连 LLM / embedding 的 HTTP 客户端 |
| `neo4j` | `>=5.28,<7` | 图库驱动（领域三元组与结构化召回） |
| `llama-index-core` | `>=0.14,<0.15` | RAG 管道骨架：Document / Pipeline / Retriever |
| `llama-index-vector-stores-qdrant` | `>=0.10,<0.11` | 向量库适配（低于此版本会 import 已被删掉的符号） |
| `qdrant-client` | `>=1.17,<1.18` | 向量库驱动，minor 与 Qdrant 服务端最多差 1 |

### 2.3 外部服务与模型

| 依赖 | 版本 / 形态 | 作用 | 谁来起 |
| --- | --- | --- | --- |
| Qdrant | `qdrant/qdrant:v1.17.1` | 向量库：语义召回 | `docker compose up -d` |
| Neo4j | `neo4j:5.26-community` | 图库：领域三元组与结构化召回 | `docker compose up -d` |
| LLM（OpenAI 兼容） | 由 `.env` 指定 | 专家评审 + 方案撰写 | 外部 API |
| Embedding 模型 | 2048 维（由 `.env` 指定） | 语料与 query 向量化 | 外部 API |

> 检索后端、模型与维度都是**配置项**，不是代码常量：改 embedding 维度必须重建
> 向量集合（`--recreate`），换语料或换 embedding 模型后必须重跑阈值标定，
> 见 [`docs/rag.md`](docs/rag.md) §5/§10。

### 2.4 开发依赖

`dev` extra（`pytest`、`ruff`），以 extra 而非 dependency-group 声明，
保证旧版 pip 也装得上：

```bash
python -m pip install -i https://pypi.org/simple -e ".[dev]"
```

---

## 3. 启动方式

### 3.1 一次性准备

```bash
# 1) 独立环境（示例用 conda，venv 等价）
conda create -n ec-renew python=3.12 -y
conda activate ec-renew

# 2) 依赖。若默认 pip 镜像不通，必须显式指定官方源
python -m pip install -i https://pypi.org/simple -e ".[dev]"

# 3) 提交前钩子（一次性）
python scripts/install_hooks.py

# 4) 配置：复制模板后填写。密钥只存在于 .env，绝不进版本库
cp .env.example .env
```

`.env` 里有两类东西：**密钥**（LLM 与 embedding 各一个）和**可调参数**
（检索后端、`top_k`、共识阈值、轮次上限等）。变量名与默认值见
[`.env.example`](.env.example)——那里面只有占位值，没有任何真实密钥。

### 3.2 离线跑通（不需要网络、不需要密钥、不需要 Docker、不需要语料）

```bash
python -m ec_renew.interface.cli --offline -r "301分段FR36污水井更换加厚板，涉及焊接，需确认合规性"
```

离线模式用确定性假模型 + 内存检索，用于确认框架闭环本身是通的。

### 3.3 真实链路

```bash
# 1) 起存储（Qdrant 向量库 + Neo4j 图库）
docker compose up -d
docker compose ps          # Qdrant: 16333/16334   Neo4j: 7474/7687

# 2) 自备语料：格式见 data/README.md，路径由 .env 的 DATA_DIR / CORPUS_FILE 决定
python -m ec_renew.rag.ingest

# 3) 标定向量阈值（换语料 / 换 embedding 模型后必做）
python -m ec_renew.rag.calibrate

# 4) 对话
python -m ec_renew.interface.cli
```

### 3.4 CLI 用法

| 命令 | 行为 |
| --- | --- |
| `python -m ec_renew.interface.cli` | 交互对话，提示符 `变更请求>` |
| `... --request/-r "<请求>"` | 单次请求后退出 |
| `... --offline` | 假模型 + 内存检索，不访问网络与存储 |
| `... --rag {auto,llamaindex,graph,memory}` | 指定检索后端（默认取 `.env` 的 `RAG_BACKEND`） |

交互模式下输入 `/exit`（`/quit`、`exit`、`quit` 同义）退出。

### 3.5 启动时的常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `pip` 装不上 | 默认镜像可能不通，显式加 `-i https://pypi.org/simple` |
| 导入到的包版本不对 | Anaconda 的用户级 `site-packages` 可能排在 env 之前并**遮蔽** env 内的同名包；用 `PYTHONNOUSERSITE=1` 启动，或 `conda env config vars set PYTHONNOUSERSITE=1 -n <env>` |
| 存储连不上、`docker port` 为空 | 宿主端口可能落在 Windows 保留区间；换 `.env` 里的端口（默认已避开 6333） |
| 无历史案例 | 属于**显式降级**：会打印原因、请求人工确认，并把保证等级降为 `knowledge_based` |

---

## 4. 质量闸门

推送前三条必须全绿（CI 与 `pre-commit` 钩子都会跑）：

```bash
python -m ruff check src tests scripts
python -m pytest tests -q
python scripts/check_secrets.py
```

无 Docker / 无语料时，相关集成用例**自动 skip 而非失败**。注意本地绿不等于 CI 绿：
`EXE001`（脚本可执行位）与依赖组解析在 Linux 上才会暴露，细节见
[`AGENTS.md`](AGENTS.md) §1.3。

---

## 5. 代码结构

> 下表路径相对 `src/ec_renew/`。

| 文件 | 职责 |
| --- | --- |
| `config.py` | 唯一读环境变量处；API key 以 `SecretStr` 持有，业务层拿不到 |
| `contracts.py` | 全部跨层 schema（外环与内环的唯一交接面） |
| `ports.py` | `LLM` / `Retriever` / `EvidenceRegistry` 接口 + `RunContext` |
| `llm.py` | `FakeLLM`（离线确定性）+ 真实 LLM 适配器 + 响应缓存 |
| `errors.py` | 异常体系；第三方异常只在适配层翻译一次 |
| `observability.py` | JSONL 事件日志（`sort_keys=True`，保证重放一致） |
| `agents/memory.py` | 内容寻址的 `EvidenceRegistry` + 确定性 `MemoryService`（Read/Filter/Project） |
| `session.py` | 多轮会话状态（Run 无状态 / Session 有状态） |
| `rag/graph.py` | `Neo4jRetriever`（领域 Cypher，唯一 Cypher 出处之一）+ `InMemoryRetriever` |
| `agents/experts.py` | 六类专家（L0）+ 规则表选专家 + prompt 构造 + 契约重试 |
| `agents/runtime.py` | `AgentRuntime`：元/子智能体**共用**的 think-act-observe 执行入口；`StepRecord` 逐 step 落盘 |
| `agents/guard.py` | 守卫：六个 act 的逐条判据，全局状态的**唯一写者**（扩基线 / `freeze` / `register` / 投影 / 额度） |
| `agents/meta.py` | 元智能体的**决策部分**（当前是规则骨架）+ `ActivationPlan` 的生产者 |
| `agents/skills/` | **技能包**（目录形式）：`expert_review/` 带 `SKILL.md` + prompt/personas/runner，六个专家复用同一技能，差别只在 persona |
| `workflow.py` | 外环：前段准备（意图 / 检索 / 落地等级）→ 调用元智能体循环 → 后段收尾（渲染 / 保证等级 / 振荡记录） |
| `interface/cli.py` | 对话式命令行 |

### `rag/` —— RAG 管道（LlamaIndex）

| 文件 | 职责 |
| --- | --- |
| `corpus.py` | 语料解析：一单一 Document，字段结构化 + `disciplines` 打标 |
| `embeddings.py` | 真实 embedding 适配器 / `FakeEmbedding`（离线确定性） |
| `llm_bridge.py` | `LlamaLLMBridge`：把项目的 `LLMPort` 接到 LlamaIndex 的 `CustomLLM` |
| `extractors.py` | `DomainTripletExtractor`：确定性领域三元组（标准 `kg_nodes`/`kg_relations`） |
| `graph_store.py` | Neo4j 领域 schema：建约束 / 写三元组 / 清空 |
| `vector_store.py` | Qdrant 集合与 payload 索引的显式创建 |
| `ingest.py` | `IngestionPipeline` 编排 + 命令行 |
| `retriever.py` | `LlamaIndexRetriever`：混合检索（RRF），实现 `RetrieverPort` |
| `factory.py` | 检索后端选择与显式降级报告 |

---

## 6. 依赖方向

```
interface → workflow → agents / rag → ports → contracts

interface → rag.factory → rag.retriever → rag.graph（复用 Cypher 与部门→专业映射）
```

`contracts` 不依赖任何业务模块；`workflow` 只见 `agents` 的抽象接口；
`rag` 不 import `workflow` / `agents` / `interface`，因此不存在循环依赖。
这条方向由 `tests/test_layering.py` 守护。

---

## 7. 已实现

### 7.1 外环与内环

- **外环固定拓扑 + 中段受限循环**：前段/后段的阶段顺序可枚举；中段是元智能体的**封闭动作空间**（六个 act，不存在自由工具调用）
- **元智能体是循环对象**：run 级常驻的 think-act-observe（D-70）；子智能体是它的一个 act，由同一个 `AgentRuntime` 执行
- **act 是请求，守卫是唯一写者**：扩基线 / `freeze()` / `register()` / 写 `ProjectionRecord` / 扣额度只发生在 `agents/guard.py`（D-71）
- **逐 step 可回放**：每次 think / act / guard / llm 都**在执行前**落一条 `step`、执行后落 `step_done`
- **分发通道零自由文本**：`dispatch_experts` 的载荷键是闭集，多一个键就整单作废（D-75）
- **有界自主性**：专家一律 L0（单次调用 + JSON 输出），不依赖 provider 的 tool calling

### 7.2 评审与共识

- **专家评审是可复用技能**：方法（评审准则 + 契约 + 重试/修复/弃权）在 `agents/skills/expert_review/`，六个专家只是同一技能 + 不同 persona（D-92）
- **判断可以基于领域通识**：本轮证据给不出技术细节时专家仍须给判断（revise/reject + 待补清单），`abstain` 只留给「超出专业范围 / 请求无法判定」
- **Delphi 式两轮共识**：第 1 轮完全隔离；第 2 轮仅披露**匿名 claim**（不含身份与完整论证）
- **弃权分两类**：判断性弃权是**交付**，执行失败弃权才是**缺席**；quorum 只拦后者（D-93）
- **共识分母修正**：分母为**配置权重和**，abstain 计 0，避免「缺席抬分」

### 7.3 收敛与交付

- **三个收敛状态，只回答「交付还是交人」**：`approved` 交付 / `manual_review` 交人裁定 / `stalled` 流程已停（D-82/D-96）
- **「因为什么交人」是独立字段**：`quorum` / `disagreement` / `conditions_only` / `evidence_gap`，报告里有「交人原因」一行（D-96）
- **判定分两层**：是否收敛（结构化规则）与停止时如何分类（交付 / 交人）分开；共识分降为**描述性支持度**（D-95）
- **保证等级看实际依据**：每条意见声明 `basis`（evidence / knowledge / mixed），声明 knowledge 的权重占比 > 0.5 时降为 `knowledge_based`
- **缺口清单是结构化产出**：由专家的 `uncertainties`/`constraints` 确定性派生成 `EvidenceRequest`，进 `RunResult` 与报告专节（D-94）
- **方案有确定性回落**：模型撰写失败时用结构化字段拼出合格方案，并显式标注「未经模型」

### 7.4 记忆、证据与可复现

- **证据可追溯**：每个意见都带 `evidence_ids`，且必须 ⊆ 本轮 registry；任何事实性结论都能回溯
- **记忆读权限在元智能体**：子 agent 无独立存储，上下文由 `MemoryService.project()` 确定性投影
- **投影是确定性函数**：`project()` 不是 LLM tool
- **可复现**：内容寻址证据 ID + 排序归并 + 响应缓存；平行分支结果按**排序**归并，禁用 LLM 摘要压缩证据

### 7.5 检索与降级

- **混合检索**：Qdrant 语义召回 + Neo4j 结构化召回，RRF 融合，无阈值调参
- **阈值有据可依**：向量阈值由 `calibrate` 在标注 query 上实测标定（域内外可分空隙取中点），不靠直觉
- **降级显式**：检索后端每次回退都带原因，打印并写事件日志；显式指定后端时绝不偷偷降级
- **无历史降级**：无历史案例 → 前置 HITL → 输出标记 `knowledge_based`

### 7.6 安全

- **密钥不外泄**：key 只在 `config.py` 读成 `SecretStr`，不进 `repr` / `model_dump` / 事件日志；`scripts/check_secrets.py` 同时在**提交前**与 CI 扫描
- **单向依赖**：见 §6，由测试守护，`contracts` 不 import 业务模块

---

## 8. 尚未实现 / 未接线

> 顺序已定（D-84）：**先 harness，后 kernel**。已完成的部分见 §7，以下均为**未完成**项。

### 8.1 未实现

- **kernel**：断点续跑、预算熔断、挂起恢复 —— 全部未实现
- **工具调用**：L1 / L2 自主层未实现，专家一律 L0
- **证据请求的检索端**：请求的渠道、词表派生、限额、幂等记账与结构化产出都已落地，但 `RetrieverPort.search()` **未实现**，故请求显式记为 `evidence_request_unsatisfied`
- **后置 HITL 的交互**：`ask_human` 只登记待办，真正挂起要等 kernel
- **OTel** 与**级联检测实验**：未开始

### 8.2 已落地但未接线

- **元智能体的 LLM 决策层**：激活集 / 权重 / 证据子集目前全部由确定性规则产出（`agents/meta.py`），「LLM 补差集」那一层未接，因此**行为与改造前等价**
- **`run_input.session`（会话快照）**：契约里已有，但规则骨架用不到 L1，**尚无消费者**；子 agent 看不到会话历史这一点已由 `tests/test_session.py` 的哨兵用例钉住

### 8.3 待标定

- **分歧归因阈值**（检测与记录已落地，Q-22）
- **停滞 / 振荡阈值**（Q-17）

> 本项目明确区分「已验证」与「未验证」：没跑过真机的路径就写「未验证」，
> 不因为代码写完就标成完成。填 [`docs/rag.md`](docs/rag.md) §9 的表格时保持同样的诚实。

---

## 9. 参与开发

开工前先读 [`AGENTS.md`](AGENTS.md)：分支模型、Conventional Commits、PR 规则、
三条质量闸门，以及「密钥绝不进版本库」的硬性约束；README 自身的维护要求也写在
那里（§8）。
