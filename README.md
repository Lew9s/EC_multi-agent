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

- **外环固定拓扑 + 中段受限循环**：前段/后段的阶段顺序可枚举；中段是元智能体的**封闭动作空间**（六个 act，不存在自由工具调用）
- **元智能体是循环对象**：run 级常驻的 think-act-observe（D-70）；子智能体是它的一个 act，由同一个 `AgentRuntime` 执行（§3.1 的唯一执行入口）
- **act 是请求，守卫是唯一写者**：扩基线 / `freeze()` / `register()` / 写 `ProjectionRecord` / 扣额度只发生在 `agents/guard.py`；元智能体只能提交提案（D-71）
- **逐 step 可回放**：每次 think / act / guard / llm 都**在执行前**落一条 `step`、执行后落 `step_done`（§9.4 的 `StepRecord`，kernel 的前置 seam）
- **分发通道零自由文本**：`dispatch_experts` 的载荷键是闭集，多一个键就整单作废（D-75）
- **专家评审是可复用技能**：方法（评审准则 + 契约 + 重试/修复/弃权）在 `agents/skills/expert_review/`，六个专家只是同一技能 + 不同 persona（D-92）
- **判断可以基于领域通识**：本轮证据给不出技术细节时专家仍须给判断（revise/reject + 待补清单），`abstain` 只留给「超出专业范围 / 请求无法判定」（D-92）
- **保证等级看实际依据**：每条意见声明 `basis`（evidence / knowledge / mixed），声明 knowledge 的权重占比 > 0.5 时保证等级降为 `knowledge_based`——检索命中历史 ≠ 专家用了它
- **弃权分两类**：判断性弃权（模型说「我无法结论」）是**交付**，执行失败弃权（超时/契约耗尽）才是**缺席**；quorum 只拦后者（D-93）
- **四种收敛状态**：`approved` 交付 / `manual_review` 分歧交人工 / `stalled` 流程已停 / `insufficient_evidence` **证据不足 → 补证据**（带结构化缺口清单，D-93）
- **缺口清单是结构化产出**：由专家的 `uncertainties`/`constraints` 经固定词表确定性派生成 `EvidenceRequest`，进 `RunResult.evidence_requests` 与报告专节（D-94）
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

## 尚未实现 / 未接线的部分

> **顺序已定**（`docs/design.md` D-84）：**先 harness，后 kernel**。

**harness（`AgentRuntime`）—— 已落地（D-89…D-91）**：元智能体是 run 级常驻的 think-act-observe
循环（D-70）；act 空间是封闭枚举的六个动作，**act 是请求而守卫是全局状态的唯一写者**（D-71）；
元/子智能体共用同一个 `AgentRuntime`（§3.1）；`ActivationPlan` 首次有了生产者与消费者；
每次 think / act / guard / llm 在执行前落 `step`、执行后落 `step_done`（§9.4）。

> ⚠ **决策部分目前只有规则骨架**：激活集 / 权重 / 证据子集全部由确定性规则产出
> （`agents/meta.py`），Q-06 三明治里「LLM 补差集」那一层**未接**。因此行为与改造前**等价**：
> 既有 124 条用例全绿，离线端到端的证据 id 逐条相同。
>
> 仍然没有消费者的输入：`run_input.session`（会话快照）。它的合法读者是元智能体的 L1 层，
> 而规则骨架用不到 L1——**接 LLM 决策时才需要**，届时一并接上。子 agent 看不到会话历史这一点
> 已由 `tests/test_session.py` 的哨兵用例钉住。

**kernel —— 未实现**：断点续跑 / 预算熔断 / 挂起恢复。

**其余未实现或未接线** —— 工具调用（L1/L2）· 分歧归因的**阈值标定**（检测与记录已落地，Q-22）·
证据请求的**检索端**（渠道、词表派生、限额、幂等记账、事件与结构化产出都已落地，但 `Retriever.search()` 未实现，故请求显式记为 `evidence_request_unsatisfied`，见 `docs/rag.md` §9）· **后置 HITL 的交互**（`ask_human` 只登记待办，
真正挂起要等 kernel）· 停滞/振荡阈值的标定（Q-17）· OTel · 级联检测实验
