# 工程变更方案生成系统 · 实验版 Demo

workflow（外环，确定性）+ agent（内环，有界自主）的多智能体方案生成系统，
RAG 管道用 LlamaIndex 搭建。
设计文档见 [`docs/design.md`](docs/design.md)，RAG 管道见 [`docs/rag.md`](docs/rag.md)。

> **参与开发前请先读 [`AGENTS.md`](AGENTS.md)** —— 它规定了分支模型、提交信息、
> PR 规则、三条质量闸门，以及「密钥绝不进版本库」的硬性约束。
>
> 一次性准备：
>
> ```bash
> python -m pip install -i https://pypi.org/simple -e ".[dev]"   # 装依赖
> python scripts/install_hooks.py                                # 装提交前钩子
> ```
>
> 每次提交前三条闸门必须全绿：
>
> ```bash
> python -m ruff check src tests scripts
> python -m pytest tests -q
> python scripts/check_secrets.py
> ```

## 快速开始

> **语料不随仓库分发**（真实业务数据，见 [`data/README.md`](data/README.md)）。
> 第 1、2 步不需要语料即可跑通；第 5 步之前请自备语料。

```bash
# 1) 离线跑通（不需要网络、不需要 key、不需要 docker、不需要语料）
python -m ec_renew.interface.cli --offline -r "301分段FR36污水井更换加厚板，涉及焊接，需确认合规性"

# 2) 跑测试（无语料/无 docker 时，相关用例自动 skip 而非失败）
python -m pytest tests -q

# 3) 起存储（Qdrant 向量库 + Neo4j 图库）
docker compose up -d

# 4) 填 key：DEEPSEEK_API_KEY（LLM）+ ZHIPU_API_KEY（智谱 embedding-3）
cp .env.example .env

# 5) 按 data/README.md 的格式自备语料（默认路径 data/zahuo.txt），然后摄取
python -m ec_renew.rag.ingest

# 6) 标定向量阈值（换语料/换模型后必做，详见 docs/rag.md §10）
python -m ec_renew.rag.calibrate

# 7) 接真实模型
python -m ec_renew.interface.cli -r "301分段FR36污水井更换加厚板"
```

## 依赖安装

本机默认的 pip 镜像（清华）不通，装依赖要显式指定官方源：

```bash
python -m pip install -i https://pypi.org/simple -e ".[dev]"
```

## 代码结构

> 下表路径相对 `src/ec_renew/`。

| 文件 | 职责 |
| --- | --- |
| `config.py` | 唯一读环境变量处；API key 以 `SecretStr` 持有，业务层拿不到 |
| `contracts.py` | 全部跨层 schema（外环与内环的唯一交接面） |
| `ports.py` | `LLM` / `Retriever` / `EvidenceRegistry` 接口 + `RunContext` |
| `llm.py` | `FakeLLM`（离线确定性）+ `DeepSeekLLM`（httpx 直连）+ 响应缓存 |
| `errors.py` | 异常体系；第三方异常只在适配层翻译一次 |
| `observability.py` | JSONL 事件日志（`sort_keys=True`，保证重放一致） |
| `agents/memory.py` | 内容寻址的 `EvidenceRegistry` + 确定性 `MemoryService`（Read/Filter/Project） |
| `session.py` | 多轮会话状态（Run 无状态 / Session 有状态） |
| `rag/graph.py` | `Neo4jRetriever`（领域 Cypher，唯一 Cypher 出处之一）+ `InMemoryRetriever` |
| `agents/experts.py` | 六类专家（L0）+ 规则表选专家 + prompt 构造 + 契约重试 |
| `workflow.py` | 主流程 `run()`：意图 → 检索 → 选专家 → 两轮 Delphi → 渲染 |
| `interface/cli.py` | 对话式命令行 |

### `rag/` —— RAG 管道（LlamaIndex）

| 文件 | 职责 |
| --- | --- |
| `corpus.py` | 语料解析：一单一 Document，字段结构化 + `disciplines` 打标 |
| `embeddings.py` | `ZhipuEmbedding`（智谱 embedding-3）/ `FakeEmbedding`（离线确定性） |
| `llm_bridge.py` | `LlamaLLMBridge`：把项目的 `LLMPort` 接到 LlamaIndex 的 `CustomLLM` |
| `extractors.py` | `DomainTripletExtractor`：确定性领域三元组（标准 `kg_nodes`/`kg_relations`） |
| `graph_store.py` | Neo4j 领域 schema：建约束 / 写三元组 / 清空 |
| `vector_store.py` | Qdrant 集合与 payload 索引的显式创建 |
| `ingest.py` | `IngestionPipeline` 编排 + 命令行 |
| `retriever.py` | `LlamaIndexRetriever`：混合检索（RRF），实现 `RetrieverPort` |
| `factory.py` | 检索后端选择与显式降级报告 |

## 依赖方向

```
interface → workflow → agents / rag → ports → contracts

interface → rag.factory → rag.retriever → rag.graph（复用 Cypher 与部门→专业映射）
```

`contracts` 不依赖任何业务模块；`workflow` 只见 `agents` 的抽象接口；
`rag` 不 import `workflow` / `agents` / `interface`，因此不存在循环依赖。
这条方向由 `tests/test_layering.py` 守护。

## 已实现的机制

- **外环固定拓扑**：阶段顺序可枚举，不存在 LLM 决定的分支
- **有界自主性**：专家一律 L0（单次调用 + JSON 输出），不依赖 provider 的 tool calling
- **Delphi 式两轮共识**：第 1 轮完全隔离；第 2 轮仅披露**匿名 claim**（不含身份与完整论证）
- **记忆读权限在元智能体**：子 agent 无独立存储，上下文由 `MemoryService.project()` 确定性投影
  （措辞见 `docs/design.md` D-70：是**读权限**，不是「所有权」——没有任何 agent 持有存储）
- **证据可追溯**：每个意见都带 `evidence_ids`，且必须 ⊆ 本轮 registry
- **无历史降级**：无历史案例 → 前置 HITL → 输出标记 `knowledge_based`
- **共识分母修正**：分母为**配置权重和**，abstain 计 0，避免"缺席抬分"
- **可复现**：内容寻址证据 ID + 排序归并 + 响应缓存
- **混合检索**：Qdrant 语义召回 + Neo4j 结构化召回，RRF 融合，无阈值调参
- **阈值有据可依**：`VECTOR_MIN_SCORE` 由 `calibrate` 在 36 条标注 query 上实测标定（域内外可分空隙 0.3390~0.3551，取中点 0.3472），不靠直觉
- **降级显式**：检索后端每次回退都带原因，打印并写事件日志；显式指定后端时绝不偷偷降级
- **密钥不外泄**：key 只在 `config.py` 读成 `SecretStr`，不进 `repr` / `model_dump` / 事件日志（含真实 key 的实测扫描）

## 尚未实现（v2）

> **顺序已定**（`docs/design.md` D-84）：**先 harness，后 kernel**。kernel 的挂起/恢复与预算熔断
> 要求 agent 执行状态可序列化，harness 须先用 `StepRecord`（§9.4）把这个 seam 留出来。

**harness（`AgentRuntime`）** —— 元智能体从「固定阶段」改为 **run 级常驻的 think-act-observe 循环**
（D-70）；act 空间封闭枚举、**act 是请求而守卫是唯一写者**（D-71）；元/子智能体共用一份运行时，
派发子智能体是元智能体的一个工具（D-72）。

> 当前实现里元智能体的**决策部分完全不存在**：`select_experts()` 是纯关键词规则、`complete_intent()`
> 从不调用 LLM（`purpose="intent"` 只被 `FakeLLM` 的分支认识）、`ActivationPlan` 是一张**零生产者
> 零消费者**的死契约、`run_input.session` **全库无人读取**。文档层面已定稿，见 `design.md` §5.4
> 与 D-70…D-85。

**kernel** —— 断点续跑 / 预算熔断 / 挂起恢复。

**其余** —— 分歧归因 · 工具调用（L1/L2）· 证据请求通道 · 轮次冻结的完整实现（当前每轮复用同一基线）·
停滞与不动点检测 · 振荡检测的阈值标定 · OTel · 级联检测实验 · `Retriever.search()`
（agent 工具侧双接口，见 `docs/rag.md` §9）
