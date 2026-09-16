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
python -m ec_renew.cli --offline -r "301分段FR36污水井更换加厚板，涉及焊接，需确认合规性"

# 2) 跑测试（无语料/无 docker 时，相关用例自动 skip 而非失败）
python -m pytest tests -q

# 3) 起存储（Qdrant 向量库 + Neo4j 图库）
docker compose up -d

# 4) 填 key：DEEPSEEK_API_KEY（LLM）+ ZHIPU_API_KEY（智谱 embedding-3）
cp .env.example .env

# 5) 按 data/README.md 的格式自备语料（默认路径 data/zahuo.txt），然后摄取
python -m ec_renew.ingest

# 6) 标定向量阈值（换语料/换模型后必做，详见 docs/rag.md §10）
python -m ec_renew.rag_llama.calibrate

# 7) 接真实模型
python -m ec_renew.cli -r "301分段FR36污水井更换加厚板"
```

## 依赖安装

本机默认的 pip 镜像（清华）不通，装依赖要显式指定官方源：

```bash
python -m pip install -i https://pypi.org/simple -e ".[dev]"
```

## 代码结构

| 文件 | 职责 |
| --- | --- |
| `config.py` | 唯一读环境变量处；API key 以 `SecretStr` 持有，业务层拿不到 |
| `contracts.py` | 全部跨层 schema（外环与内环的唯一交接面） |
| `ports.py` | `LLM` / `Retriever` / `EvidenceRegistry` 接口 + `RunContext` |
| `llm.py` | `FakeLLM`（离线确定性）+ `DeepSeekLLM`（httpx 直连）+ 响应缓存 |
| `errors.py` | 异常体系；第三方异常只在适配层翻译一次 |
| `observability.py` | JSONL 事件日志（`sort_keys=True`，保证重放一致） |
| `memory.py` | 内容寻址的 `EvidenceRegistry` + 确定性 `MemoryService`（Read/Filter/Project） |
| `session.py` | 多轮会话状态（Run 无状态 / Session 有状态） |
| `rag.py` | `Neo4jRetriever`（领域 Cypher，唯一 Cypher 出处之一）+ `InMemoryRetriever` |
| `experts.py` | 六类专家（L0）+ 规则表选专家 + prompt 构造 + 契约重试 |
| `workflow.py` | 主流程 `run()`：意图 → 检索 → 选专家 → 两轮 Delphi → 渲染 |
| `cli.py` | 对话式命令行 |

### `rag_llama/` —— LlamaIndex RAG 管道

| 文件 | 职责 |
| --- | --- |
| `corpus.py` | 语料解析：一单一 Document，字段结构化 + `disciplines` 打标 |
| `embeddings.py` | `ZhipuEmbedding`（智谱 embedding-3）/ `FakeEmbedding`（离线确定性） |
| `llm.py` | `LlamaLLMBridge`：把项目的 `LLMPort` 接到 LlamaIndex 的 `CustomLLM` |
| `extractors.py` | `DomainTripletExtractor`：确定性领域三元组（标准 `kg_nodes`/`kg_relations`） |
| `graph_store.py` | Neo4j 领域 schema：建约束 / 写三元组 / 清空 |
| `vector_store.py` | Qdrant 集合与 payload 索引的显式创建 |
| `ingest.py` | `IngestionPipeline` 编排 + 命令行 |
| `retriever.py` | `LlamaIndexRetriever`：混合检索（RRF），实现 `RetrieverPort` |
| `factory.py` | 检索后端选择与显式降级报告 |

## 依赖方向

```
cli → workflow → experts / rag → ports → contracts
                    └── memory ──┘

cli → rag_llama.factory → rag_llama.retriever → rag（复用 Cypher 与部门→专业映射）
```

`contracts` 不依赖任何业务模块；`workflow` 只见 `agents` 的抽象接口；
`rag.py` 不 import `rag_llama`，因此不存在循环依赖。

## 已实现的机制

- **外环固定拓扑**：阶段顺序可枚举，不存在 LLM 决定的分支
- **有界自主性**：专家一律 L0（单次调用 + JSON 输出），不依赖 provider 的 tool calling
- **Delphi 式两轮共识**：第 1 轮完全隔离；第 2 轮仅披露**匿名 claim**（不含身份与完整论证）
- **记忆所有权在元智能体**：子 agent 无独立存储，上下文由 `MemoryService.project()` 确定性投影
- **证据可追溯**：每个意见都带 `evidence_ids`，且必须 ⊆ 本轮 registry
- **无历史降级**：无历史案例 → 前置 HITL → 输出标记 `knowledge_based`
- **共识分母修正**：分母为**配置权重和**，abstain 计 0，避免"缺席抬分"
- **可复现**：内容寻址证据 ID + 排序归并 + 响应缓存
- **混合检索**：Qdrant 语义召回 + Neo4j 结构化召回，RRF 融合，无阈值调参
- **阈值有据可依**：`VECTOR_MIN_SCORE` 由 `calibrate` 在 36 条标注 query 上实测标定（域内外可分空隙 0.3390~0.3551，取中点 0.3472），不靠直觉
- **降级显式**：检索后端每次回退都带原因，打印并写事件日志；显式指定后端时绝不偷偷降级
- **密钥不外泄**：key 只在 `config.py` 读成 `SecretStr`，不进 `repr` / `model_dump` / 事件日志（含真实 key 的实测扫描）

## 尚未实现（v2）

自研 kernel（断点续跑/预算熔断/挂起恢复）· 分歧归因 · 工具调用（L1/L2）· 证据请求通道 ·
轮次冻结的完整实现（当前每轮复用同一基线）· OTel · 级联检测实验 · `Retriever.search()`
（agent 工具侧双接口，见 `docs/rag.md` §9）
