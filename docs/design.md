# 工程变更方案生成系统 · 设计文档

> **状态**：讨论中（首次固化版）
> **用途**：留痕与校验。文中每项决策标注 `[已定]` / `[待定]` / `[建议]`
> **范围**：架构原则、模块划分、主流程拓扑、核心机制、契约草案、异常与上下文管理
> **不含**：具体实现代码、阈值参数取值、实验设计

---

## 0. 背景

### 0.1 既有实现的问题（重写依据）

| 问题 | 证据 |
| --- | --- |
| 密钥硬编码 | `llamaindex.py` / `app.py` / `add_groupkey.py` / `readme.md` 中均为 `password="12345678"` |
| 代码节点静默失败 | Dify `计算共识达成率` 节点 `except: return 0`，把「解析失败」等同于「无共识」 |
| 无语法校验关卡 | Dify `生成循环变量` 节点含全角冒号（`"eval_xizhuang"："None"`），实测 `SyntaxError: invalid character '：' (U+FF1A)` |
| 命名不一致 | 同一概念混用 `xizhuang` / `electronic` / `lunji` 与中文专业名 |
| 模块级副作用 | `llamaindex.py` / `RAG.py` 在 import 时读文件、建图、调用 `nest_asyncio.apply()`，无 `__main__` 守卫 |
| 依赖版本分裂 | `ec` 使用 pydantic 1.10 + llama-index 0.10；`CDIACR` 使用 pydantic 2.10 + langgraph |
| 提交不可追溯 | `ec` 历史提交为 `update` × N，无法定位变更范围 |
| 意图识别单点失效 | `CDIACR/normalize_request` 仅凭三个关键词判定变更类型 → 决定激活专家 → 决定全部后续 |
| 共识分母错误 | `CDIACR/consensus_gate` 的 `weight_sum` 只累加「在场意见」的权重，专家缺席反而抬高共识分 |

### 0.2 旧工作流规模

Dify DSL `变更方案生成.yml`：**56 节点 / 76 边**
（llm 16、assigner 19、if-else 12、code 3、loop 1、loop-start 1、http-request 1、parameter-extractor 1、start 1、answer 1）

### 0.3 领域对象

来自既有图谱 schema：
- 实体：`CHANGE_ORDER` / `COMPONENT` / `DEPARTMENT` / `REASON` / `TIME_POINT`
- 关系：`MODIFIES` / `SIGNED_BY` / `HAS_REASON` / `OCCURS_AT` / `PART_OF`

专家标识固定为六类（来源：旧工作流 `代码执行` 节点的 `EXPERT_ORDER`）：

| ID | 专业 |
| --- | --- |
| E01 | 结构设计 |
| E02 | 舾装工艺 |
| E03 | 质量规范 |
| E04 | 电气系统 |
| E05 | 轮机系统 |
| E06 | 材料与焊接 |

---

## 1. 架构原则

| # | 原则 | 说明 |
| --- | --- | --- |
| P1 | **外环确定、内环有界自主** | workflow 管阶段顺序与终止；agent 管推理方式 |
| P2 | **强类型交接** | 外环与内环之间只通过 pydantic schema 交互 |
| P3 | **事实断言必须可回溯** | 任何事实性结论必须引用 `evidence_id`，无例外 |
| P4 | **不能自主判断时交给人，不猜** | 缺少事实基础时进入 HITL，而非强行产出 |
| P5 | **降级必须显式** | 禁止静默降级；降级须写入事件日志并在输出中可见 |
| P6 | **异常只在边界翻译一次** | 上层不得见到第三方库的异常类型 |
| P7 | **可复现** | 同输入可重放；随机性、阈值、策略均需快照记录 |

---

## 2. 技术选型

| 项 | 选择 | 状态 | 备注 |
| --- | --- | --- | --- |
| 异步模型 | asyncio | `[已定]` | 相关陷阱见 §7.1 |
| 编排引擎 | **自研内核**（不用 LangGraph） | `[已定]` | 理由见 §2.1 |
| Checkpoint | JSONL 事件日志 | `[已定]` | 兼作可解释性数据源 |
| Tracing | OpenTelemetry（厂商中立） | `[已定]` | 不绑定 LangSmith |
| 类型 | pydantic v2 | `[已定]` | 统一 `ec` / `CDIACR` 的分裂 |
| 代码规范 | ruff（line-length 100, py312）+ pytest | `[建议]` | 沿用 `CDIACR` 配置 |
| RAG 框架 | **LlamaIndex**（`IngestionPipeline` + `VectorStoreIndex`） | `[已定]` | 见 [`rag.md`](rag.md) |
| LLM | DeepSeek `deepseek-flash`（httpx 直连，OpenAI 兼容） | `[已定]` | 保留自研 `DeepSeekLLM`（缓存/用量/异常翻译），经 `CustomLLM` 桥接给 LlamaIndex |
| 向量模型 | 智谱 `embedding-3`（默认 2048 维） | `[已定]` | OpenAI 兼容 `/embeddings`；支持 2048/1024/512/256 |
| 向量库 | Qdrant（docker compose） | `[已定]` | 集合由本项目显式创建，维度必须与 embedding 对齐 |
| 图库 | Neo4j（docker compose） | `[已定]` | 保留 §0.3 的领域 schema，不改既有 Cypher |
| 混合检索融合 | Reciprocal Rank Fusion | `[已定]` | 只用排名，规避「余弦 vs 匹配计数」的量纲问题 |

### 2.1 为什么不用 LangGraph

`CDIACR` 中 LangGraph 的实际使用范围经核查仅为：
`StateGraph(Model)` / `add_node` / `set_entry_point` / `add_edge` / `add_conditional_edges` / `compile()` / `.invoke()`。

以下能力**全部未使用**：checkpoint（`MemorySaver` 等）、并行扇出（`Send`）、人工中断（`interrupt`）、消息归约（`add_messages`）。

即：现有代码把 LangGraph 当作「约 120 行的手写图执行器」使用，其真正有价值的能力一项未用。

同时存在一个隐患：现有节点以**原地改写**方式更新状态（`state.change_type = ...`），该模式在串行图中可用，但一旦引入并行扇出即产生竞态。

结论：自研内核的替换成本远低于直觉；需要自建的四项能力（checkpoint / 扇出 / 中断 / 归约）中，仅 checkpoint 成本较高，而 JSONL 事件日志 + 重放即可覆盖。

---

## 3. 模块组成

| # | 模块 | 职责 | 允许依赖 | 明确不做 |
| --- | --- | --- | --- | --- |
| 0 | `kernel/` | 图执行引擎：调度、扇出、归约、预算、事件、取消、挂起/恢复 | 无 | 不含任何业务语义 |
| 1 | `contracts/` | 全部跨模块 schema（唯一接口层） | 无 | 不依赖任何业务模块 |
| 2 | `ports/` | `LLM` / `Retriever` / `GraphStore` / `EvidenceRegistry` / `Tool` 接口 + 第三方异常翻译 | `contracts` | 不含实现 |
| 3 | `config/` | 配置、模型与索引版本矩阵、策略快照 | 无 | 不允许被业务模块绕过 |
| 4 | `observability/` | OTel span + JSONL 事件写入 | `contracts` | 不决定 span 语义 |
| 5 | `rag/` | 摄取、图谱 schema、检索（双接口） | `ports`, `contracts` | 不做共识、不认识专家 |
| 6 | `workflow/` | **外环**：阶段、拓扑、归约器、约束守卫 | `agents`(仅接口面), `rag`, `kernel` | 不写 prompt |
| 7 | `agents/` | **内环**：专家注册表、运行时、工具、persona、人类决策解析 | `ports`, `contracts` | 不改变拓扑、不写库 |
| 8 | `integration/` | Neo4j / Ollama 具体实现（唯一适配层） | `ports` | 不含业务逻辑 |
| 9 | `interface/` | CLI / API / 对话呈现 / 导出 | `workflow` | 不做编排 |

### 3.1 依赖方向

```
interface → workflow → agents / rag → ports → contracts
                │                    │
                └──────── kernel ────┘        integration 仅被 ports 的装配点引用
```

**硬规则**：
- 只允许单向向下依赖；`rag` 不得 import `workflow`，`contracts` 不得 import 任何业务模块。
- Neo4j 驱动只允许出现在 `rag/` 与 `integration/`。
- `workflow` 只允许看到 `agents` 的抽象接口（`AgentRuntime`），不得 import 具体专家实现。
- LLM 输出不得直接进入 `eval` / `exec`。

---

## 4. 主流程拓扑

```
用户提问（可能高度抽象）
  │
  ▼
normalize ── 规则优先抽取实体；LLM 兜底判定变更类型
  │
  ▼
intent_complete ── 规则抽取 + 图谱展开 + LLM 补全 → 多路 query
  │
  ▼
Pass 1  probe retrieval（广度：探明涉及哪些维度）
  │
  ▼
assess_grounding ── 是否存在真实历史案例？
  │
  ├──[有历史]────────────────────────────────────────────────┐
  │                                                          │
  │   freeze（冻结本轮证据基线）                              │
  │      ▼                                                   │
  │   meta_agent ── 产出 ActivationPlan（激活集+权重+证据子集）│
  │      ▼                                                   │
  │   Pass 2  targeted retrieval（按激活集定向补全）           │
  │      ▼                                                   │
  │   dispatch（并行扇出至专家 agent）                        │
  │      ▼                                                   │
  │   aggregate → consensus_gate                              │
  │      ▼                                                   │
  │   未达标 → attribute_disagreement → revise_intent → 回 freeze（迭代）
  │                                                          │
  └──[无历史]────────────────────────────────────────────────┘
      │
      ▼
   meta_agent 产出 HumanReviewRequest
      │
      ▼
   ⏸ checkpoint 挂起（写 JSONL + resume_token）
      │
      ▼
   用户：接受默认判断 / 提供自己的见解
      │
      ▼
   provided_facts → Evidence(source="human")，原文保留
      │
      ▼
   meta_agent 重入 → ActivationPlan → 分配子专家
      │
      ▼
   单轮评估（用户提供实质信息则可迭代）→ 标记 knowledge_based
      │
      ▼
   finalize / human_review
```

**拓扑性质**：阶段顺序固定、可枚举、可在部署期校验（无非法跳转、每个循环有出口、可达性检查）。意图与查询集是**状态**而非输入。
---

## 5. 核心机制

### 5.1 有界自主性

外环与内环的职责边界：

| 维度 | Workflow（外环） | Agent（内环） |
| --- | --- | --- |
| 控制什么 | 阶段顺序、是否迭代、何时终止、是否人工介入 | 如何得出结论、看哪些证据、调哪些工具 |
| 拓扑 | 固定、可枚举、部署期校验 | 不固定、运行时自主 |
| 输出 | State patch | `ExpertOpinion`（强类型） |
| 预算 | 整轮：轮次上限、总 token、墙钟 | 单专家：步数、token、单次超时 |
| 失败影响面 | 全局（终止 / 转人工） | 局部（`abstain` / 仅重跑该专家） |
| 审计粒度 | 每阶段一条事件 | 每个决策点一条事件 |
| 可复现性 | 完全确定 | 有界（temp=0 + 工具白名单 + 步数上限 + 轨迹回放） |

**一句话**：agent 决定「怎么想」，workflow 决定「什么时候想、几个人想、想几轮、算不算数」。

#### 四条硬边界

| # | 边界 | 内容 |
| --- | --- | --- |
| B1 | 拓扑边界 | agent 不得改变 workflow：不得自行开启新一轮、不得激活其它专家、不得跳过共识判定 |
| B2 | 工具边界 | 仅只读白名单工具；任何写入一律经外环的 proposal → approval 通道 |
| B3 | 预算边界 | 单专家步数 / token / 超时上限；超限返回**部分结论 + `partial=true`**，不报错 |
| B4 | 输出边界 | 必须返回 `ExpertOpinion`，且 `evidence_ids` 非空、可回溯 |

#### 自主度分档（每专家可配）

| 档位 | 行为 | 适用 |
| --- | --- | --- |
| L0 | 单次调用 + 结构化输出 | 纯评判型（如质量合规） |
| L1 | 允许 1–2 次工具调用（自主补检索） | 多数专家（**建议默认**） |
| L2 | 完整 think → act → observe 循环 + 自我校验 | 高不确定场景（如结构方案设计） |
| ~~L3~~ | 专家自行派生下级专家 | **禁止**：成本与审计失控 |

> 灵活性体现在「每个专家的自主度可配置」，而非「所有专家全部放开」。

### 5.2 证据机制

#### 5.2.1 EvidenceRegistry（run 级核心组件）

任何检索——无论外环或 agent 发起——都必须登记回 registry 并分配 `evidence_id`。

- 使约束（`evidence_ids ⊆ 本轮检索集`）始终成立，集合以**登记方式扩展**而非被绕过
- 事件日志可追溯「哪个专家补充检索了什么」
- 可统计「共享基线 / 各自补充」比例（实验指标）

实现方式：`EvidenceRegistry` 作为 `RunContext` 的一个端口（与 `budget`、`events` 并列），接口定义在 `ports/`，使其对 `workflow` 与 `agents` 同时可用而不引入相互依赖。

证据来源枚举：

```python
EvidenceSource = Literal["graph", "text", "human", "standard"]
```

#### 5.2.2 RAG 双接口

```
                    ┌──────────── rag/ ────────────┐
                    │ 同一混合检索管道 + 同一 schema │
                    └──────┬──────────────┬────────┘
              prefetch()   │              │   search()
         （外环·确定·一次）  │              │  （agent tool·自主·多次）
                           ▼              ▼
                   EvidenceBundle    EvidenceRefs
                           └──────► EvidenceRegistry ◄──────┘
                                （run 级·唯一登记点）
```

| | `prefetch()` | `search()` |
| --- | --- | --- |
| 发起者 | workflow `retrieve` 阶段 | 具体专家（L1/L2） |
| 次数 | 每 run 一次 | 受配额限制 |
| 作用 | 共享事实基线 + 门禁判定 | 补充探索 |
| 可复现 | 完全确定 | 靠轨迹回放 |

**为何不能只把 RAG 作为 tool**：共识的前提是专家在评估同一批事实。若检索完全由 agent 自主发起，`consensus_score` 会变成「不同信息条件下的不同判断」的加权平均，方法论上无效。

**门禁语义**：只对 `prefetch()` 的基线结果判定，**不计入 agent 补充的证据**，否则门禁结果依赖随机行为，不可复现。

#### 5.2.3 Tool 的四项配套机制

| # | 机制 | 内容 |
| --- | --- | --- |
| 1 | 配额 | 单专家最多 N 次检索（建议 2）；全局上限（建议 12）；复用连接并发上限 |
| 2 | 缓存 | `query → result` 缓存，避免重复 embedding 与重复图查询 |
| 3 | 返回值 | tool 只返回 `EvidenceRef`（`evidence_id` + 一句话摘要），**全文留在 registry** |
| 4 | 轨迹回放 | 每次 tool 调用（参数 + 返回的 `evidence_ids`）写入事件日志；重放时按记录回放，**不重新检索** |

第 3 点是为控制 agent 上下文增长——L2 循环中若直接返回证据全文，数轮即撑爆上下文。

#### 5.2.4 轮次级证据冻结

**目标**：任何一轮内，所有意见都建立在一个被冻结、且所有专家共享的事实基线之上。

```
Round N:
  prefetch → 冻结基线 EvidenceSet_N → 所有专家共享（各自看子集）
      │
   专家推理（仅使用 EvidenceSet_N 内的证据）
      │
   产出 ExpertOpinion + EvidenceView_N
      │
   发现关键证据缺失 → 提交 EvidenceRequest
      │
   外环裁定 → 成立则在 Round N+1 的基线上扩（所有人同时看到新证据）
```

关键点：**agent 不得单方面扩大自己的事实基础**，只能报告缺口。

#### 5.2.5 解释性检索 vs 事实性检索

| 类型 | 例子 | 是否引入新事实主张 | 处理 |
| --- | --- | --- | --- |
| 解释性 | 查规范条文、术语定义、组件层级 | 否 | 允许轮内自主，计入 `delta_ids` |
| 事实性 | 查历史变更记录、相似案例 | 是 | **必须**走 `EvidenceRequest` |

理由：规范条文属共享背景知识；历史案例是新的事实主张，一旦不同专家引入不同案例，讨论即不可比。

建议以**两个不同工具**固化该边界（`search_evidence` / `request_evidence`），避免依赖 agent 自行判断。

#### 5.2.6 分歧归因

共识未达成时，`attribute_disagreement` 必须先给分歧定性，再决定补救路径：

| 归因 | 判定依据 | 补救路径 |
| --- | --- | --- |
| `judgment`（判断分歧） | 证据重叠高、结论相反 | 只重跑冲突专家（意图与 query 不变） |
| `evidence`（证据分歧） | 证据重叠低，且差异证据与结论相关 | **不重跑专家**；先扩大冻结基线，下一轮所有人可见 |
| `mixed` | 两者兼有 | 先补证据，再重跑冲突专家 |

**这是双接口缺口的真正补法**：不是让所有专家看到相同内容，而是能判断分歧来自「看的不一样」还是「想的不一样」。

#### 5.2.7 专家集变更规则

- **系统自动修订**：专家集只增不减（静默移除会掩盖问题）
- **用户主动调整**：不受此限，但必须留痕（`excluded_by_user` + 原话）
- 新增专家看到的是**当前轮冻结基线**，与其他专家公平
- 共识计算需能标注「第 N 轮加入」，否则历史轮次分数不可比

### 5.3 意图演化

#### 5.3.1 定位调整

| | 旧定位 | 新定位 |
| --- | --- | --- |
| 决定什么 | 变更类型 → 专家集 → 权重 → **事实基线** | 只决定**候选专家集** + 初始 query |
| 精度要求 | 必须正确 | **允许不完整**（错误可恢复） |
| 性质 | 定论 | **假设**，显式标记 `provenance` |
| 失败后果 | 全局错误且不可察觉 | 候选集偏小 → 后续阶段暴露并补齐 |

理由：跨域相关性只有领域专家知道。通用意图识别智能体在**信息位置上**就无法完成跨端子问题拆解，非提示词优化可解。故把**发现职责交给专家，裁定职责留给外环**。

#### 5.3.2 graph-grounded 意图补全

用图谱编码的协作知识补全抽象提问。既有 Cypher 已在执行该路径（组件 → 历史变更组 → 签收部门），现将其从「查历史」升级为「意图补全的输入」：

```
用户："FR36 想加个扶强筋"
  ├─ 规则抽取：实体 FR36
  ├─ 图谱展开：所属父结构 / 历史变更组 / 历史签收部门
  │            → 「历史相关变更涉及：船体结构、焊接、质检」
  └─ LLM 补全：子问题 + 多路 query
```

#### 5.3.3 意图为状态而非输入

```python
class IntentState(BaseModel):
    revision: int
    change_type: str
    sub_questions: list[SubQuestion]
    active_experts: list[str]
    weights: dict[str, float]
    query_set: list[str]                      # query 集同样随轮演化
    provenance: Literal["rule", "llm", "expert", "disagreement"]

class IntentRevision(BaseModel):
    round: int
    from_revision: int
    to_revision: int
    trigger: Literal["evidence_request", "disagreement", "cross_domain_flag"]
    changes: list[str]                        # 人类可读：改了什么
    evidence_ids: list[str]                   # 凭什么改（必须可回溯）
```

原则：**修订允许，静默修订禁止**。每次修订须有 `trigger` 与支撑证据。

边界：`max_sub_questions` / `max_active_experts` / `max_intent_revisions`；`plan` 阶段需含优先级裁剪，防止证据爆炸。

### 5.4 元智能体（meta_agent）

#### 5.4.1 输入：带结构元信息的证据

若只把证据全文交给元智能体，它需自行语义判断「这些证据属于哪个专业」——退回猜测。故检索结果须携带结构化专业归属：

```python
class EvidenceMeta(BaseModel):
    evidence_id: str
    entity_kinds: list[str]      # COMPONENT / DEPARTMENT / REASON / TIME_POINT
    entity_keys: list[str]
    disciplines: list[str]       # ← 从图谱关系推导的专业归属（关键字段）
    group_keys: list[str]        # 变更单分组
    score: float
```

`disciplines` 的来源：`COMPONENT → 历史变更组 → SIGNED_BY 部门 → 专业`。
`[待定]` 推导时机：(a) 检索时实时推导 / **(b) 建图阶段预先打标（建议）**。

#### 5.4.2 输出

```python
class ActivationPlan(BaseModel):
    active_experts: list[str]
    weights: dict[str, float]                # 校验 Σ=1，只保留 >0
    evidence_scope: dict[str, list[str]]     # 专家 → evidence_id 子集
    rationale: str
    cross_domain_flags: list[str]            # 兜底通道
```

**硬约束**：`evidence_scope` 中所有 id 必须存在于 registry。元智能体**不得编造 `evidence_id`**。

#### 5.4.3 激活策略

`[建议]` 规则 + LLM 混合：`disciplines` 命中达阈值的专业由**规则直接激活**（可复现），LLM 只处理规则未覆盖的部分。每条激活须带 `rationale` 与引用的 `evidence_ids`。

#### 5.4.4 cross_domain_flags（兜底通道）

首轮 Pass 1 若漏掉某维度，元智能体看不到它、也就不会激活对应专家（「漏专家 → 漏证据 → 漏请求者」闭环）。故保留该通道：专家共识轮内任一专家均可标记「本改动还涉及 X 域，建议邀请 E0x」。

理由：一旦专家真正开始推理，其对跨域影响的判断远比元智能体盲判可靠。


#### 5.4.5 职责拆分：Memory Controller

元智能体的职责不是「router」，而是 **Memory Controller**，由两部分组成：

| 部分 | 性质 | 内容 |
| --- | --- | --- |
| 决策部分 | 可含 LLM（规则优先） | 激活集、权重、证据子集分配 |
| `MemoryService` | **确定性代码** | `Read` / `Filter` / `Project` |

```
M_global --Read--> M_retrieved --Filter--> M_filtered --Project--> C_i
```

其中 `Project` 产出 `ExpertTask`，是外环向子 agent **分发事实的唯一通道**。详见 §8.4 / §8.5。

#### 5.4.6 投影必须是确定性的（不作 LLM tool）

**判断准则**（通用）：

> 输入空间**开放**（需语义理解才能构造参数）→ 做成 tool。
> 输入空间**封闭**（参数可直接从状态推出）→ 做成确定性函数。

据此对照两类操作：

| | RAG 检索 | 上下文投影 |
| --- | --- | --- |
| 输入 | 自然语言 query（**开放**） | `(expert, round, disclosure_policy)`（**封闭枚举**） |
| 自主性价值 | 高 | **零** |
| 操作性质 | **扩充**事实 | **分发**事实 |
| 出错后果 | 证据多/少，**可见、可登记、可纠正** | 事实基础被**静默裁剪**，不可见 |
| 可复现性 | 轨迹回放即可 | **必须确定性** |

结论：RAG 做 tool 正确（D-11）；**投影做 tool 错误**。

若投影由 LLM 自由调用，会同时破坏三件事：

1. **首轮独立性**——LLM 可自行选择 `disclosure=anon_claims`，绕过策略
2. **可复现性**——同输入不同投影
3. **防注入边界**——重新打开循环注入通道（§8.5.3）

**若未来必须 tool 化**，需同时满足三条件：参数全为枚举、无自由文本；`disclosure` 由 `round` 经策略推出且 LLM 不可覆盖；投影结果须通过三项断言（不含他人身份、`evidence_ids ⊆ registry`、`hard_constraints` 完整）。

**Read 保留为未来的 tool 位置**：记忆规模变大后，元智能体「读取全局记忆某个视图」的输入空间是半开放的，可作为 tool。`MemoryService` 的接口按 tool 形状设计，便于后续包装。

### 5.5 无历史分支与 HITL

#### 5.5.1 二分支降级

```
prefetch → 是否存在真实历史案例？
    ├── 有 → 正常共识流程（冻结基线 → 迭代 → 方案）
    └── 无 → 元智能体产出 HumanReviewRequest → 用户判断 → 再分配专家
```

> 设计取舍：曾考虑 4 级 sufficiency（full / partial / thin / none），但其对应的**动作类别只有两种**，级别不驱动不同动作即无意义，故简化为布尔二分支。

间接案例：组件本身无历史时，向上查其父结构（`PART_OF`）的历史案例，标记 `source: indirect`，归入「有历史」分支。成本为检索层多一条 Cypher，可显著扩大覆盖面。`[建议保留]`

#### 5.5.2 前置 HITL 契约

```python
class HumanReviewRequest(BaseModel):
    reason: Literal["no_history"] = "no_history"
    understood_request: str                  # 系统理解到的请求
    retrieved_evidence: list[EvidenceRef]    # 检索到了什么（可能为空）
    missing: list[str]                       # 缺什么
    candidate_experts: list[str]             # 系统倾向（原先验规则降级为推荐）
    questions: list[str]                     # 需要用户回答的问题

class HumanDecision(BaseModel):
    proceed: bool                            # 是否继续
    approved_experts: list[str]
    extra_experts: list[str] = []
    excluded_by_user: list[str] = []         # 用户明确排除 + 原话
    provided_facts: list[str] = []           # 用户补充信息
    note: str = ""
```

呈现要求：必须同时给出**系统理解、检索结果、不确定项**，否则用户无判断依据。

#### 5.5.3 用户输入即证据

用户提供的补充信息须登记为 `Evidence(source="human")` 并分配 `evidence_id`，使约束（`evidence_ids` 非空且可回溯）**无需破例**，审计链保持完整。

**原文保留要求**：若解析层将用户表述结构化，须原文与解析结果并存：

```python
class HumanProvidedFact(BaseModel):
    raw_text: str                # ← 用户原话，不可改写
    parsed: ParsedFact | None    # ← 解析结果，可为空
```

#### 5.5.4 对话式交互与解析层

```
系统消息：[初步判断 + 候选专家 + 请确认]
    ├── 用户点【接受】 → 直接构造 HumanDecision（无需解析）
    └── 用户自由表述   → 解析层 → HumanDecision → 回显确认 → 继续
```

必须回显确认：把「不用看质量」解析为「移除 E03」与「降低 E03 权重」是不同结果，执行错误将浪费整轮。

#### 5.5.5 迭代条件

| 用户输入 | 轮次 |
| --- | --- |
| 仅接受默认判断（无新信息） | **单轮** |
| 提供实质信息（`provided_facts` 非空） | 允许迭代 |

理由：无新信息时迭代只会让模型互相附和，不产生新信息（「共识幻觉」）。

#### 5.5.6 必须防止的误读

> **用户「接受默认判断」不等于方案变可信。**

用户确认的是**评估范围**，不是**结论**。即使用户接受，输出仍标记 `knowledge_based`（无历史依据）。否则用户会误以为「我确认过 → 方案有保障」。

#### 5.5.7 前置 HITL 与后置 manual_review 的区分

| | 前置 HITL | 后置 manual_review |
| --- | --- | --- |
| 时机 | 分配专家**之前** | 出结论**之后** |
| 目的 | 补信息、定范围 | 审核结论 |
| 触发 | 无历史证据 | 共识未达标 / 高风险 / 保证等级不足 |
| 输入 | `HumanDecision` | approve / reject / modify |

两者为**不同状态字段**（如 `pending_human_scope` / `pending_human_review`），不得共用布尔值，否则恢复时无法判断分支。

### 5.6 共识计算

```python
class Disagreement(BaseModel):
    experts: tuple[str, str]
    kind: Literal["judgment", "evidence", "mixed"]
    evidence_overlap: float          # 两专家证据集的 Jaccard 重叠度
    divergent_evidence: list[str]    # 仅出现在其中一方的 evidence_id
```

**分母修正（必须）**：共识分母为本轮**激活专家的配置权重和**，而非「在场意见」的权重和。
`abstain` 按 0 分计入分母。

> 修正依据：`CDIACR/consensus_gate` 现有实现的分母只包含在场意见，含义是「专家缺席 → 分母变小 → 共识分变高」，即「证据越少越容易达成共识」，方向相反。

**部分失败策略**（quorum + 逐专家降级）：

| 项 | 规则 |
| --- | --- |
| 失败专家 | 产出 `abstain` 意见（而非缺席） |
| 权重 | 从分母扣除 |
| 下限 | 有效专家数 < 3 → 整轮失败 |
| 建议 | 扇出使用收集模式（`return_exceptions=True`），因 LLM 调用已付费 |

### 5.7 保证等级

```python
class AssuranceLevel(BaseModel):
    level: Literal[
        "history_backed",     # 有历史证据支撑
        "knowledge_based",    # 仅基于规范或通用工程知识
    ]
    supporting_evidence: list[str]   # history_backed 时非空
    inference_basis: list[str]       # 迁移或先验的依据
```

**硬约束**：最终方案中任何条目不得「无支撑」。若出现无支撑条目，不得直接交付，强制人工复核。

若有全部专家 `abstain` 的情形，系统应输出「证据不足以评估」，而非强行给出方案。
---

## 6. 契约清单（字段级草案）

以下为 `contracts/` 的 schema 草案，是外环与内环之间唯一的交接面。字段为设计层面定义，尚未定稿为实现。

### 6.1 证据类

```python
EvidenceSource = Literal["graph", "text", "human", "standard"]

class Evidence(BaseModel):
    evidence_id: str                     # 全局唯一，由 EvidenceRegistry 分配
    source: EvidenceSource
    content: str                         # 原文，不可被系统改写
    entity_kinds: list[str] = []
    entity_keys: list[str] = []
    disciplines: list[str] = []          # 专业归属
    group_keys: list[str] = []           # 变更单分组
    source_file: str = ""
    source_offset: int = 0
    score: float = 0.0
    created_at: datetime

class EvidenceRef(BaseModel):            # 给 agent 的轻量引用（控制上下文）
    evidence_id: str
    gist: str                            # 一句话摘要
    source: EvidenceSource

class EvidenceMeta(BaseModel):           # 给 meta_agent 的结构化元信息
    evidence_id: str
    entity_kinds: list[str]
    entity_keys: list[str]
    disciplines: list[str]
    group_keys: list[str]
    score: float
    source: EvidenceSource

class EvidenceBundle(BaseModel):         # prefetch() 的返回
    round: int
    baseline_ids: list[str]
    items: list[EvidenceMeta]
    warnings: list[str]                  # 含 low_evidence 等门禁信号

class EvidenceView(BaseModel):           # 单个专家在本轮看到的视图
    round: int
    baseline_ids: list[str]              # 本轮冻结基线（全员一致）
    delta_ids: list[str] = []            # 该专家专属增量（解释性检索）
    requested_ids: list[str] = []        # 该专家请求但未纳入的证据

class EvidenceRequest(BaseModel):        # agent → workflow（事实性检索请求）
    expert: str
    query: str
    reason: str                          # 为什么现有证据不足
    scope: Literal["components", "departments", "cases", "standards"]
```

### 6.2 意图类

```python
class EntityRef(BaseModel):
    name: str
    kind: Literal["COMPONENT", "DEPARTMENT", "REASON", "TIME_POINT", "UNKNOWN"]
    in_graph: bool
    graph_key: str | None = None

class GraphExpansion(BaseModel):         # graph-grounded 补全结果
    parent_components: list[str] = []
    historical_departments: list[str] = []
    historical_disciplines: list[str] = []
    similar_case_ids: list[str] = []

class SubQuestion(BaseModel):
    text: str
    discipline: str                      # E01..E06
    priority: float

class EvidenceNeed(BaseModel):
    description: str
    scope: Literal["components", "departments", "cases", "standards"]

class IntentCompletion(BaseModel):
    raw_request: str
    normalized_request: str
    mentioned_entities: list[EntityRef]
    graph_expansion: GraphExpansion
    sub_questions: list[SubQuestion]
    query_set: list[str]                 # 多路改写后的 query
    provenance: Literal["rule", "graph", "llm"]

class IntentState(BaseModel):
    revision: int
    change_type: str
    sub_questions: list[SubQuestion]
    active_experts: list[str]
    weights: dict[str, float]
    query_set: list[str]
    provenance: Literal["rule", "llm", "expert", "disagreement"]

class IntentRevision(BaseModel):
    round: int
    from_revision: int
    to_revision: int
    trigger: Literal["evidence_request", "disagreement", "cross_domain_flag"]
    changes: list[str]
    evidence_ids: list[str]              # 修订依据，必须可回溯
```

### 6.3 激活与人机

```python
class ScopeProposal(BaseModel):          # 保留：专家侧跨域信号
    expert: str
    cross_domain_flags: list[str]

class ActivationPlan(BaseModel):
    active_experts: list[str]
    weights: dict[str, float]            # Σ=1，只保留 >0
    evidence_scope: dict[str, list[str]] # 专家 → evidence_id 子集
    rationale: str
    cross_domain_flags: list[str] = []

class HumanReviewRequest(BaseModel):
    reason: Literal["no_history"] = "no_history"
    understood_request: str
    retrieved_evidence: list[EvidenceRef]
    missing: list[str]
    candidate_experts: list[str]
    questions: list[str]

class HumanProvidedFact(BaseModel):
    raw_text: str                        # 用户原话，不可改写
    parsed: ParsedFact | None = None     # 解析结果，可为空

class HumanDecision(BaseModel):
    proceed: bool
    approved_experts: list[str]
    extra_experts: list[str] = []
    excluded_by_user: list[str] = []
    provided_facts: list[HumanProvidedFact] = []
    note: str = ""
```

### 6.4 专家与共识

```python
ReviewDecision = Literal["approve", "revise", "reject", "abstain"]

class ExpertTask(BaseModel):             # 外环 → 内环
    expert: str
    request: str
    sub_questions: list[SubQuestion]
    evidence_view: EvidenceView
    evidence: list[EvidenceRef]          # 仅该专家子集
    autonomy: Literal["L0", "L1", "L2"]
    budget: AgentBudget
    round: int

class ExpertOpinion(BaseModel):          # 内环 → 外环
    expert: str
    decision: ReviewDecision
    rationale: str
    evidence_ids: list[str]              # 非空，且 ⊆ registry
    risk_level: Literal["low", "medium", "high"] = "low"
    recommendations: list[str] = []
    assurance: AssuranceLevel
    partial: bool = False                # 因预算超限而截断
    tool_trace: list[ToolCall] = []       # 供轨迹回放

    @computed_field
    @property
    def score(self) -> float:
        return {"approve": 1.0, "revise": 0.5, "reject": 0.0, "abstain": 0.0}[self.decision]

class Disagreement(BaseModel):
    experts: tuple[str, str]
    kind: Literal["judgment", "evidence", "mixed"]
    evidence_overlap: float
    divergent_evidence: list[str]

class AssuranceLevel(BaseModel):
    level: Literal["history_backed", "knowledge_based"]
    supporting_evidence: list[str] = []
    inference_basis: list[str] = []

class PolicySnapshot(BaseModel):         # 冻结策略，随 run 记录
    model_config = ConfigDict(frozen=True)
    weights: dict[str, dict[str, float]]
    consensus_threshold: float
    max_rounds: int
    min_effective_experts: int = 3
    sufficiency_thresholds: dict[str, float] = {}
```
---


### 6.5 会话与记忆类

```python
class TurnSummary(BaseModel):
    turn: int
    user_request: str                    # 用户原话
    conclusion: str                      # 结构化摘要（非 LLM 生成）
    confirmed_experts: list[str]
    evidence_ids: list[str]              # 引用，不含全文
    assurance: AssuranceLevel
    open_threads: list[str] = []

class SessionSnapshot(BaseModel):        # 传给 run 的不可变副本
    model_config = ConfigDict(frozen=True)
    session_id: str
    anchor_request: str                  # 首轮原话，永不摘要（防漂移）
    recent_turns: list[TurnSummary]      # 滑动窗口
    user_constraints: list[str] = []

class SessionState(BaseModel):           # 会话本体，仅由会话层修改
    session_id: str
    anchor_request: str
    recent_turns: list[TurnSummary] = []
    user_constraints: list[str] = []
    pending_run_id: str | None = None    # 有挂起 run 时不再开启新 run

    def snapshot(self) -> SessionSnapshot: ...
    def append(self, delta: TurnSummary) -> None: ...

class RunInput(BaseModel):               # run 只拿到快照副本
    request: str
    session_snapshot: SessionSnapshot

class RunResult(BaseModel):
    conclusion: str
    evidence_ids: list[str]
    assurance: AssuranceLevel
    turn_delta: TurnSummary              # 交还给会话层的唯一产物
```

### 6.6 跨 Agent 投影类

```python
class Claim(BaseModel):                  # 跨 agent 传播的最小单位
    claim: str = Field(max_length=200)   # 短句，防注入；原样复制不改写
    condition: str | None = None         # 条件不可省
    evidence_ids: list[str] = Field(min_length=1)
    discipline: str                      # 来源领域（投影时用于过滤自己）

class DisclosurePolicy(str, Enum):
    NONE = "none"                        # 首轮：完全隔离
    ANONYMOUS_CLAIMS = "anon_claims"     # 后续轮：仅匿名 claim

    @classmethod
    def for_round(cls, round: int) -> DisclosurePolicy:
        # 策略由 round 推出，不由 LLM 决定
        return cls.NONE if round == 0 else cls.ANONYMOUS_CLAIMS

class CrossAgentInfo(BaseModel):
    anonymous_claims: list[Claim] = []   # 字段选择，原样复制
    hard_constraints: list[str] = []     # 不可被 Filter 丢弃
    consensus_score: float = 0.0
    dissent_count: int = 0

class ReviewFeedback(BaseModel):         # Delphi 式受控反馈
    round: int
    consensus_score: float
    dissent_count: int
    anonymous_dissent: list[str] = []    # 不含身份与完整论证

class RevisionContext(BaseModel):        # mode == "revise" 时注入
    own_previous: ExpertOpinion          # 必须是该专家本人的
    new_evidence_ids: list[str] = []
    feedback: ReviewFeedback

class ProjectionRecord(BaseModel):       # 审计：本轮向谁披露了什么
    round: int
    expert: str
    policy: DisclosurePolicy
    disclosed_claim_ids: list[str]
    hard_constraint_ids: list[str]
```

`ExpertTask` 修订（新增 `mode` / `revision` / `cross_agent` 与归属断言）：

```python
class ExpertTask(BaseModel):
    mode: Literal["initial", "revise"] = "initial"
    expert: str
    request: str
    sub_questions: list[SubQuestion]
    evidence_view: EvidenceView
    evidence: list[EvidenceRef]
    autonomy: Literal["L0", "L1", "L2"]
    budget: AgentBudget
    round: int
    revision: RevisionContext | None = None      # mode == "revise" 时必填
    cross_agent: CrossAgentInfo | None = None    # 首轮为 None

    @model_validator(mode="after")
    def _check_revision_ownership(self):
        if self.revision is not None:
            assert self.revision.own_previous.expert == self.expert, \
                "own_previous 必须属于该专家本人"
        return self
```

`ExpertOpinion` 修订（新增四个字段）：

```python
    claims: list[Claim] = []             # 可跨 agent 传播的最小单位
    constraints: list[str] = []          # 硬约束（不可被 Filter 丢弃）
    uncertainties: list[str] = []        # 不确定项
    confidence: float = Field(default=0.0, ge=0, le=1)   # 仅记录，不参与计分
```

## 7. 异常处理

### 7.1 最重要的边界

> **可预期的业务结果走状态，异常只留给「意外」。**

判定依据：`共识未达成` / `证据不足` / `专家弃权` **均不是异常**，是状态字段。

反例（旧实现）：Dify `计算共识达成率` 用 `except: return 0` 把「JSON 解析失败」混进业务通道，使解析错误被当作「无共识」，直接污染决策。

### 7.2 异常分类

```python
class ECError(Exception): ...

# 可重试
class TransientError(ECError):          # 超时 / 连接重置 / 限流 / 池耗尽
    retry_after: float | None = None

# 不可重试，但可降级
class ContractViolation(ECError):       # LLM 输出不满足 schema
    node: str; attempt: int; raw_excerpt: str

# 不可重试，直接失败
class PermanentExternalError(ECError):  # 认证失败 / 索引不存在 / 权限不足
class InvalidRequest(ECError):          # 空问题 / 非法组件名 → 400

# 内部 bug，必须 fail fast
class InvariantViolation(ECError):      # 归约器违反结合律 / 非法状态跳转
class BudgetExceeded(ECError)
class StepLimitExceeded(ECError)
```

**关键约束**：`InvariantViolation` **永远不被捕获**。它代表引擎或状态机写错，重试无意义。此规则用于防止「用重试掩盖并发 bug」。

### 7.3 各层职责

| 层 | 允许捕获 | 必须向上抛 | 异常翻译责任 |
| --- | --- | --- | --- |
| `ports` / adapter | **第三方异常**（`neo4j.*` / `httpx.*` / `ollama.*`） | — | **唯一**翻译点 |
| `rag` | 无 | 全部 | 否 |
| `agents` | `ContractViolation`（内部契约重试） | `InvariantViolation` | 否 |
| `workflow` 节点 | 无（编排不 try） | 全部 | 否 |
| `kernel` executor | 全部（写事件、收集兄弟结果） | 包装为 `NodeFailure` 后按策略处理 | 否 |
| `interface` | 全部 | — | 映射到退出码 / HTTP 状态 |

**核心原则**：异常只在边界翻译一次。上层代码中不得出现 `httpx.TimeoutException` 或 `neo4j.ServiceUnavailable`。

### 7.4 异常必须可序列化

因需写入 JSONL 事件日志：

```python
@dataclass(frozen=True)
class FailureEvent:
    code: str            # "contract_violation"
    node: str
    step: int
    cause_type: str      # 原始类型名，便于聚合统计
    message: str         # 受控长度
    retryable: bool
    attempt: int
```

规则：日志中**不写** traceback 全文、prompt 原文、连接串。traceback 仅进 stderr。

### 7.5 重试

- 重试发生在**端口层或节点层**，不在 kernel 层（kernel 无法判断幂等性）。
- 两类重试必须分开计数：

| 类型 | 触发 | prompt 是否变 | 幂等 |
| --- | --- | --- | --- |
| 传输重试 | `TransientError` | 否 | 是（仅多花钱） |
| 契约重试 | `ContractViolation` | **是**（回灌校验错误） | 否，产物不同 |

- 重试前必须先落事件，否则进程崩溃后会重复付费。
- 退避：指数 + 抖动；本地 Ollama 与远程服务使用两套参数。
- **输出 token 截断是契约重试的头号来源**（JSON 被切断 → 校验失败 → 重发 → 再次截断），须给输出预留窗口（见 §8.2）。

### 7.6 部分失败

| 项 | 规则 |
| --- | --- |
| 失败专家 | 产出 `abstain`，不缺席 |
| 共识分母 | 仍计入配置权重（计 0 分），不因缺席而缩小 |
| 下限 | 有效专家数 < 3 → 整轮失败 |
| 扇出模式 | 建议 `return_exceptions=True`（收集）；`InvariantViolation` 与 `CancelledError` 仍必须立刻上抛 |

### 7.7 禁止事项

- 禁止裸 `except:` / `except Exception: pass`
- 禁止 except 后返回默认值（如 `return 0`）
- 禁止在 except 中抛出无链异常（必须 `raise X from e`）
- 禁止用异常做正常控制流（循环终止用哨兵状态）
- 禁止吞掉 `asyncio.CancelledError`（见 §8.1）

---

## 8. 上下文管理

本系统中「上下文」指两件不同的事，须分开设计。

### 8.1 运行时上下文（RunContext）

```python
@dataclass(frozen=True)
class RunContext:
    run_id: str
    trace: TraceHandle        # OTel span 句柄
    budget: Budget            # 共享可变对象（见下）
    cancel: CancelToken
    events: EventSink         # JSONL 写入器
    registry: EvidenceRegistry
    ports: Ports              # LLM / Retriever / GraphStore
```

**载体策略**：以显式传参 `(State, Ctx)` 为主；`ContextVar` 仅用于跨第三方库的隐式关联（HTTP trace 注入、日志 correlation）。

理由：显式传参可测试、可静态检查；ContextVar 的隐式传播会使调试困难。

#### asyncio 的三个硬陷阱

**1. `asyncio.gather` 会拷贝 context，`set()` 不回传**

- 只读（`trace_id`、`budget` 引用）→ 安全
- 用 `ContextVar.set()` 累积 → **子任务中的 set 父任务不可见，静默丢数据**
- 累积类数据须放**共享可变对象**（如 `Budget` 实例），在无 `await` 区间内读改写（asyncio 单线程下安全）

**2. `CancelledError` 是 `BaseException`，不得被 `except Exception` 吞掉**

```python
except asyncio.CancelledError:
    raise
except Exception as e:
    ...
```

建议在 CI 中加规则：凡 `except Exception` 必须紧跟 `except asyncio.CancelledError: raise`。

**3. 取消穿不透阻塞调用**

`request_timeout=3600` 的配置意味着单次调用最长 1 小时，图级取消无法落下。**超时必须挂在 HTTP 客户端层（每次调用）**，不能只依赖 `max_steps`。

#### 扇出失败的取消语义

`asyncio.gather` 默认**不取消**兄弟任务。两种选择：

| 方案 | 含义 | 取舍 |
| --- | --- | --- |
| 收集（`return_exceptions=True`） | 等待全部完成 | **建议**：LLM 调用已付费，取回结果配合 quorum |
| 快速失败 + 取消兄弟 | 立即终止 | 省钱，但丢弃在途结果 |

收集模式下须注意：`InvariantViolation` 与 `CancelledError` 仍必须立刻上抛，否则真 bug 会被当作普通异常收进列表。

### 8.2 LLM 上下文（窗口预算）

#### 预算桶（显式配置）

```
总窗口
├─ system / persona          固定
├─ 任务指令 + 输出 schema    固定
├─ 证据（evidence）          可裁剪：按相关度 + token 预算
├─ 历史 / 上一轮结论         可压缩
└─ 输出预留                  硬预留，不可被挤占
```

输出预留是硬约束：不留输出空间 → JSON 截断 → 契约重试 → 可能死循环（与 §7.5 耦合）。

#### 证据注入

- 检索条数不按固定 `top_k`，改为**按 token 预算裁剪**
- 每条证据必须带 `evidence_id`，与可回溯约束对齐
- **每个专家只接收相关子集**（旧实现的 `E0x_context` 分发设计正确，须保留）；全量下发将使长上下文问题乘以专家数

#### 压缩策略

| 方式 | 可预测 | 成本 | 破坏 `evidence_id` 可追溯 |
| --- | --- | --- | --- |
| 截断 | 是 | 0 | 否 |
| 结构化裁剪（按字段） | 是 | 0 | 否 |
| LLM 摘要 | 否 | 高 | **是** |

**建议默认禁用 LLM 摘要**：摘要会切断 `evidence_id` 与文本的对应关系，直接违反可回溯约束。

#### 多轮迭代的上下文

不得把每一轮的完整专家意见回灌 prompt。仅回灌：
- 上一轮**结论摘要**
- **冲突点**（谁与谁不一致、争在哪）
- **被要求重跑专家的原意见**

#### 超限检测时机

必须在**发请求前**估算（tokenizer），超限时由客户端决定截断策略，而非依赖服务端报错或静默截断。

### 8.3 两者交界：证据生命周期

```
检索 → evidence_ids 集合
     → 按专家分发子集（E0x_context）
     → 专家产出意见（引用 evidence_ids）
     → 契约校验（证据必须 ⊆ registry）
     → 共识计算（分母 = 配置权重和；abstain 计 0）
     → 未达标 → 只回灌冲突 → 下一轮（round+1，有上限）
     → 达标 / 超限 → 整合 → HITL 或 finalize
```
---


### 8.4 会话与记忆（跨对话轮）

**先区分两种「多轮」**，混淆会导致设计错误：

| | 共识迭代轮 | 对话轮次 |
| --- | --- | --- |
| 范围 | run 内（第 1..N 轮） | 会话内（用户第 1..N 次提问） |
| 生命周期 | 一次请求内，结束即销毁 | **跨请求持续** |
| 覆盖情况 | §5.2 / §5.6 已定义 | 本节定义 |

#### 8.4.1 核心原则：Run 无状态，Session 有状态

> **Run 是纯函数**：输入 = 会话快照 + 当前请求 → 输出 = 结果 + 会话增量。
> **Session 是累加器**：只保存显式的、结构化的小数据，不保存 run 的内部过程。

`run()` 只拿到快照的**副本**，不持有 `SessionState` 引用 → **run 物理上改不了会话状态**。跨轮污染在结构上不可能发生，因为没有隐式通道。

两项直接收益：
- 每个 run 独立可复现（输入完全明确）
- **第 10 轮与第 1 轮的子 agent 上下文规模完全一致**

#### 8.4.2 三层上下文

| 层 | 生命周期 | 内容 | 谁看得到 |
| --- | --- | --- | --- |
| 会话上下文 | 跨对话轮 | 用户原话、已确认范围、约束、历史结论摘要 | 只有元智能体 |
| 运行上下文 | 单次 run | 证据基线、本轮意见、共识状态、分歧 | 外环 |
| 智能体上下文 | 单次 LLM 调用 | `ExpertTask` | 单个子 agent |

上下文预算按层分配，不混用：元智能体的预算里放会话摘要；**子 agent 的预算里只放本轮证据，不含任何历史**。

#### 8.4.3 唯一存储 + 投影视图

「三层记忆」中**只有全局记忆是真存储**：

| 层 | 是否真存储 | 实质 |
| --- | --- | --- |
| 全局记忆（Global Memory） | ✅ **唯一真源** | `State` + `EvidenceRegistry` + `SessionState` |
| 子 agent 私有记忆 | ❌ **投影视图** | 每次由元智能体当场构造，跑完即弃 |
| 上下文投影 | ❌ 构造过程本身 | `ExpertTask` |

若把「子 agent 私有记忆」实现成真存储，就需要同步、一致性协议与防漂移——等于把「共享记忆池」换个名字造回来。

#### 8.4.4 记忆所有权在元智能体

子 agent **内部不保存任何东西**。它的「记忆」是 `ExpertTask` 的字段，由元智能体决定给什么。

- **隔离仍在**：子 agent 没有隐式状态，不可能「偷偷」记住他人意见
- **连续性也在**：元智能体显式注入该专家自己的上次判断
- **可审计**：任务里有什么，事件日志里就有什么

**术语精确化**：子 agent 是「无**共享**状态」，而非「无状态」。若真的无状态，第二轮就是重新采样而非修订，「共识达成」会退化为「反复采样直到模型自洽」——这在方法论上不是共识，自洽也不能证明正确。

#### 8.4.5 可见性规则

| 子 agent 能看到 | 子 agent 绝不能看到 |
| --- | --- |
| 本轮请求与子问题 | 其它专家的意见（任何形式） |
| 本轮自己的证据子集 | 会话历史 |
| **自己的上一轮判断** | 历史 run 的结论 |
| 匿名聚合反馈与匿名 claim | 其它专家的身份 |
| 硬约束（hard constraints） | 其它专家的原始推理 |

#### 8.4.6 会话快照保留策略

| 内容 | 进快照？ | 形式 |
| --- | --- | --- |
| 用户原话（最近 N 轮） | ✅ | 原文 |
| 首轮原话 | ✅ | **原文，永不摘要**（锚点） |
| 已确认范围 / 专家集 | ✅ | 结构化 |
| 用户明确约束 | ✅ | 原文列表 |
| 上一轮结论 | ✅ | 结构化摘要 + `evidence_ids` |
| 上一轮的专家意见 | ❌ | **不进**（污染源） |
| 上一轮的证据全文 | ❌ | 只存 ID，需要时按 ID 取 |
| run 内部过程 | ❌ | 事件日志里有 |

**核心规则：保留引用，不保留副本。**

#### 8.4.7 摘要漂移防护

若每轮「用上次摘要生成新摘要」，误差会累积（生成式损失）。防护三条：

1. **锚点**：首轮原话永不摘要，一直保留
2. **最近 N 轮保留原文**（demo 建议 N=3）
3. 超出窗口的轮次才摘要，且**从原文生成**，不从上次摘要派生

demo 使用**结构化摘要**（从 run 结果中挑字段拼接），不调用 LLM——零成本、确定性、且不会漂移。

#### 8.4.8 一轮只允许一个活跃 run

`SessionState.pending_run_id` 非空时，新提问提示「上一个请求仍在等待你的确认」，不并发开启新 run，避免会话内状态交错。


### 8.5 跨 Agent 信息治理

#### 8.5.1 Delphi 式受控反馈

本系统的共识机制本质是 **Delphi 法**（匿名专家 + 迭代 + 受控反馈）：

| Delphi 原则 | 实现方式 |
| --- | --- |
| 匿名 | 子 agent 永不见其它专家的身份 |
| 受控反馈 | 迭代时只回灌**聚合统计与匿名 claim**，不回灌完整论证 |

**信息级联的成因**：若所有 agent 都能看到所有人的历史，则会出现

```
Agent A → high risk
Agent B 看到了 → high risk
Agent C 看到了 A+B → high risk
最终共识 = 100%，但这是从众，不是独立共识
```

因此回灌的必须是**论据**而非**结论压力**：

| ❌ 结论压力 | ✅ 论据传递 |
| --- | --- |
| 「Production Agent 认为风险很高，你重新考虑一下」 | 「另一领域提出一项约束：若设备安装先于开孔，存在可达性冲突（依据 E12）」 |

#### 8.5.2 披露规则

> **披露的最小单位是 `claim + condition + evidence`，永不披露 `judgment + 是谁`。**

| 信息 | 子 agent 可见性 |
| --- | --- |
| 其它 agent 原始对话 / 私有记忆 / 内部推理 | ❌ |
| 元智能体提取的事实 | ✅ |
| 元智能体提取的冲突点（匿名 claim） | ✅ |
| 元智能体转发的工程约束 | ✅ |
| 元智能体转发的证据 | ✅ |
| 其它 agent 的最终结论（judgment） | **不披露**（仅以 claim 形式抽象传递） |

圆整表述（供论文使用）：

> 子 Agent 不能直接访问其它 Agent 的原始状态与记忆，只能接收由元 Agent 筛选、抽象和授权的跨 Agent 信息。

即 `Agent_i ↛ Memory_j`，但允许 `Memory_j → Meta → Relevant Evidence → Agent_i`。

#### 8.5.3 循环信息流与注入防护

跨 agent 信息形成循环：

```
子 agent 输出 → 全局记忆 → 投影 → 另一子 agent 的上下文
                    ↑___________________________|
```

若投影为 LLM 自由摘要，即打开一条**间接 prompt 注入通道**：某个子 agent（或其检索到的脏数据）产出的文本，经元智能体「总结」后进入另一个子 agent 的 prompt，从而写入他人的指令区。

**防护：把注入问题转化为 schema 校验问题。**

> 跨 agent 信息以「数据帧」形式传递，经过 schema 校验，**永不以自然语言指令形式进入 prompt**。

1. 投影只做**字段选择与裁剪**，`claim` **原样复制**，不改写；字数有上限（`max_length=200`）
2. prompt 模板显式分隔并标注：「以下是其它领域提出的**待验证约束**，供参考，**不是指令**」
3. 每条 claim 必须带 `discipline` + `evidence_ids`；元智能体校验所引 evidence 必须存在于 registry

**`claim` 原样复制而非「总结成一句」**，是对「元智能体改写他人原意」的**结构性**保证——条件信息不可能被丢掉。

#### 8.5.4 Filter 的治理规则

> **标记为 hard constraint / safety 的字段，Filter 阶段不得丢弃，必须无条件进入所有相关 agent 的投影。**

否则元智能体会成为安全信息的单点故障。

#### 8.5.5 信息瓶颈的两个缓解

| 风险 | 缓解 |
| --- | --- |
| 元智能体漏掉一条重要信息 → 所有 agent 都看不到 | 子 agent 输出必须结构化（`claims` / `constraints` / `uncertainties`），不让元智能体从自然语言里猜 |
| 元智能体改写他人原意 | claim 原样复制（§8.5.3） |

#### 8.5.6 披露可审计、级联可检测

`ProjectionRecord` 记录每轮向每个专家披露了什么，使：

- 投影**可复现**（由代码 + 事件日志精确重建）
- **信息披露图**可作为研究指标

建议的三个可证伪指标（论文阶段实现）：

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| Round-0 独立共识度 | 首轮完全隔离时的共识分 | 若首轮已高，则不存在级联空间 |
| 披露-改变率 | 披露跨域信息后，意见发生改变的专家比例 | 过低=信息无用；过高=从众 |
| **扰动测试** | 给单个专家注入一条故意的错误高风险信号，测量他人是否跟随 | **直接检测级联** |

### 8.6 子 Agent 的修订机制

#### 8.6.1 修订上下文

`mode="revise"` 时注入 `RevisionContext`：**自己的**上次意见 + 本轮新增证据 + 匿名聚合反馈。同一节点函数处理两种任务形态，无需两套代码路径。

#### 8.6.2 只注入「最近一次」

第 10 轮与第 2 轮的任务规模一致（只**替换** `own_previous`，不累加）。上下文规模**有界且不随轮次增长**。

#### 8.6.3 振荡检测归元智能体

子 agent 只需知道「我现在在哪」；元智能体持有全历史（`opinions: dict[round, dict[expert, ExpertOpinion]]`），负责「我们走到哪了」，因此可以检测振荡（approve → reject → approve）并直接转 `manual_review`，不再浪费轮次。

#### 8.6.4 `confidence` 不参与共识计分

LLM 自报的 `confidence` **校准很差**（普遍过度自信）。若进入权重，会成为方法论弱点。

建议：**demo 只记录、不使用**。若研究要用，须先做校准验证（按 confidence 分桶看实际准确率）。在此之前，共识只用 `decision` 的离散分数。

## 9. 基础设施

### 9.1 JSONL 事件日志

一次 run 的全部过程事件以 JSONL 追加写入，兼作三类用途：

| 用途 | 说明 |
| --- | --- |
| Checkpoint | 崩溃 / 挂起后可重放恢复 |
| 可解释性 | 决策轨迹的数据源（可直接供既有 explainability 模块消费） |
| 实验数据 | 共识轮次、证据扩展比例、意图修订次数等指标 |

事件类型（草案）：

| 事件 | 时机 | 关键字段 |
| --- | --- | --- |
| `run_started` | run 入口 | `run_id`, `policy_snapshot`, `input_hash` |
| `node_started` | 节点执行前 | `node`, `step` |
| `node_finished` | 节点成功后 | `node`, `step`, `patch`, `tokens`, `ms` |
| `tool_called` | agent 调用工具 | `expert`, `tool`, `args`, `returned_ids` |
| `evidence_registered` | 证据入库 | `evidence_id`, `source`, `round` |
| `intent_revised` | 意图修订 | `from_revision`, `to_revision`, `trigger`, `changes` |
| `degradation` | 发生降级 | `level`, `reason`, `source_chain` |
| `suspended` | 挂起等待人工 | `reason`, `resume_token` |
| `resumed` | 恢复 | `resume_token` |
| `failure` | 节点失败 | `FailureEvent` 内容 |
| `run_finished` | run 结束 | `status`, `assurance_level` |

### 9.2 挂起与恢复

| 要求 | 说明 |
| --- | --- |
| `resume_token` | 挂起事件携带，用于定位具体 run（多会话可能同时挂起） |
| 状态可重建 | JSONL 需能重放到挂起点；节点执行前先落 `node_started`，恢复时跳过已完成节点 |
| 幂等恢复 | 否则一次 HITL 会重复调用 LLM 与检索，直接产生费用 |
| 可查询挂起列表 | 需能列出所有 `pending_human_*` 的 run |
| **无自动超时降级** | 超时保持挂起状态，不自动产出无依据的结论 |

### 9.3 OTel

- 一个 run 一个 root span；每个阶段一个 child span；每次 LLM 调用一个 span
- span 需携带 `run_id` / `node` / `round` / `expert` 属性
- span 与 JSONL 事件通过 `run_id` + `step` 关联
- **禁止**在 span 属性中写入 prompt 原文、连接串、密钥

---

## 10. 决策记录（Decision Log）

| ID | 决策 | 理由 | 状态 |
| --- | --- | --- | --- |
| D-01 | 全新重写，不基于 Dify DSL | 旧 DSL 56 节点难维护、无校验关卡、静默失败 | `[已定]` |
| D-02 | 不使用 LangGraph，自研内核 | 现有用法仅覆盖约 8 个 API，四项重能力均未使用；自研可控可审计 | `[已定]` |
| D-03 | asyncio 作为并发模型 | 6 专家并行是效率痛点的唯一解 | `[已定]` |
| D-04 | JSONL 事件日志作 checkpoint | 兼作可解释性与实验数据源 | `[已定]` |
| D-05 | OTel 厂商中立 tracing | 不绑定 LangSmith | `[已定]` |
| D-06 | workflow（外环）+ agent（内环）混合架构 | 流程标准化 + 专家灵活性兼得 | `[已定]` |
| D-07 | 两级编排 + 强类型交接 | 外环与内环唯一交接面为 pydantic schema | `[已定]` |
| D-08 | 十模块划分与单向依赖 | 防止框架与领域逻辑互相渗透 | `[已定]` |
| D-09 | 有界自主性四条边界（拓扑/工具/预算/输出） | 灵活性必须有边界，否则不可审计、成本失控 | `[已定]` |
| D-10 | 自主度分档 L0/L1/L2，每专家可配；禁用 L3 | 灵活性载体是「可配置」而非「全放开」 | `[已定]`（默认档位待定） |
| D-11 | RAG 双接口：`prefetch`（外环）+ `search`（agent tool） | 共识要求共享事实基线，纯 agentic 检索会使共识失去意义 | `[已定]` |
| D-12 | `EvidenceRegistry` 作为 run 级端口 | 使可回溯约束对两条路径同时成立且不引入模块耦合 | `[已定]` |
| D-13 | tool 返回 `EvidenceRef`（引用+摘要），全文留 registry | 控制 agent 上下文增长 | `[已定]` |
| D-14 | 轨迹回放：记录 tool 调用，重放不重算 | 使 agent 自主性不损害可复现性 | `[已定]` |
| D-15 | 门禁仅对 `prefetch` 基线判定 | 否则门禁结果依赖随机行为 | `[已定]` |
| D-16 | 轮次级证据冻结 | 保证同一轮内所有意见基于同一事实基线 | `[已定]` |
| D-17 | 区分解释性检索（轮内允许）与事实性检索（走请求通道） | 规范条文不引入新事实主张；历史案例引入 | `[已定]` |
| D-18 | 分歧归因（judgment / evidence / mixed）分路补救 | 双接口缺口的真正补法 | `[已定]` |
| D-19 | 专家集自动修订只增不减；用户调整不受限但留痕 | 静默移除会掩盖问题 | `[已定]` |
| D-20 | 意图识别定位降级为「决定候选专家集」 | 跨域相关性只有领域专家知道，通用意图体不可能完成拆解 | `[已定]` |
| D-21 | 意图与 query 为随轮演化的状态，修订须显式可审计 | 首轮 query 改写不可能一次做对 | `[已定]` |
| D-22 | graph-grounded 意图补全 | 图谱编码了「什么改动涉及哪些部门」的协作知识 | `[已定]` |
| D-23 | 首轮为两段：Pass 1 probe → meta_agent → Pass 2 targeted | Pass 1 探明维度，Pass 2 按激活集定向补全 | `[已定]` |
| D-24 | meta_agent 输入携带 `disciplines` 结构元信息 | 避免退回语义猜测 | `[已定]` |
| D-25 | meta_agent 不得编造 `evidence_id` | 可回溯约束的自然延伸 | `[已定]` |
| D-26 | 降级简化为二分支（有历史 / 无历史） | 4 级设计对应的动作类别只有两种 | `[已定]` |
| D-27 | 无历史 → 前置 HITL，用户判断后再分配专家 | 系统不猜，把判断权交给人 | `[已定]` |
| D-28 | 用户输入登记为 `Evidence(source="human")`，原文保留 | 可回溯约束无需破例，审计链完整 | `[已定]` |
| D-29 | 默认单轮；用户提供实质信息才迭代 | 无新信息时迭代只是「共识幻觉」 | `[已定]` |
| D-30 | 对话内呈现 + 解析层 + 回显确认 | 用户不接受表单式交互 | `[已定]` |
| D-31 | 前置 HITL 与后置 manual_review 为两个独立状态 | 恢复时需据此判断分支 | `[已定]` |
| D-32 | 挂起不自动超时降级 | 自动降级会产出无依据结论 | `[已定]` |
| D-33 | 业务结果走状态，异常只留给意外 | 修复 `except: return 0` 一类污染决策的写法 | `[已定]` |
| D-34 | 异常只在边界（ports）翻译一次 | 使存储/模型可替换 | `[已定]` |
| D-35 | `InvariantViolation` 永不捕获 | 防止用重试掩盖并发 bug | `[已定]` |
| D-36 | 传输重试与契约重试分开计数 | 产物不同、成本不同 | `[已定]` |
| D-37 | 部分失败：`abstain` + quorum + 最少有效专家数 3 | 避免缺席被当作通过 | `[已定]`（阈值 3 待确认） |
| D-38 | 共识分母改为「激活专家的配置权重和」 | 修复「专家缺席 → 共识分变高」的反向缺陷 | `[已定]` |
| D-39 | 运行时上下文以显式传参为主，ContextVar 仅作跨库关联 | 可测试、可静态检查 | `[已定]` |
| D-40 | 扇出采用收集模式（`return_exceptions=True`） | LLM 调用已付费 | `[建议]` |
| D-41 | 默认禁用 LLM 摘要压缩 | 摘要破坏 `evidence_id` 可追溯性 | `[已定]` |
| D-42 | 保证等级两档：`history_backed` / `knowledge_based` | 支撑「让用户决定」的信息基础 | `[已定]` |
| D-43 | Run 无状态、Session 有状态；run 只接受会话快照副本 | 跨轮污染在结构上不可能发生（无隐式通道） | `[已定]` |
| D-44 | 子 agent 无**共享**状态，记忆所有权在元智能体 | 「无状态」表述不准确，会导致共识退化为反复采样 | `[已定]` |
| D-45 | 子 agent 上下文由 `ExpertTask` 显式注入（含 `own_previous`） | 修订必须基于上次判断，而非重新采样 | `[已定]` |
| D-46 | 只注入「最近一次」自己的判断 | 上下文规模有界，不随轮次增长 | `[已定]` |
| D-47 | `own_previous.expert == expert` 归属断言 | 堵住最隐蔽的污染方式 | `[已定]` |
| D-48 | 唯一存储 + 投影视图（子 agent 无独立存储） | 避免把「共享记忆池」改名造回来 | `[已定]` |
| D-49 | Delphi 式受控反馈（匿名 + 只回灌聚合与 claim） | 防止信息级联；使共识分数在方法论上成立 | `[已定]` |
| D-50 | 披露最小单位为 claim，永不披露 judgment + 身份 | 传递论据而非结论压力 | `[已定]` |
| D-51 | 投影是确定性函数，不作 LLM tool | 否则破坏首轮独立性、可复现性与防注入边界 | `[已定]` |
| D-52 | 判断准则：输入空间开放→tool；封闭→确定性函数 | 解释了 RAG（tool）与投影（函数）的不对称 | `[已定]` |
| D-53 | Read 保留为未来 tool 位置 | 记忆规模变大后的合法 tool；demo 不需要 | `[已定]` |
| D-54 | 跨 agent 信息以数据帧传递 + schema 校验 | 把注入问题转化为可解的校验问题 | `[已定]` |
| D-55 | `claim` 原样复制，不改写，`max_length=200` | 结构性保证条件信息不被丢弃 | `[已定]` |
| D-56 | hard constraint 不得被 Filter 丢弃 | 避免元智能体成为安全信息单点故障 | `[已定]` |
| D-57 | `ExpertOpinion` 增加 `claims`/`constraints`/`uncertainties`/`confidence` | 摘要最容易丢的正是这些 | `[已定]` |
| D-58 | `confidence` 仅记录，不参与共识计分 | LLM 自报置信度校准差，会成方法论弱点 | `[已定]`（校准验证后可改） |
| D-59 | 会话快照：锚点 + 最近 N 轮原文 + 结构化摘要 | 防摘要漂移；demo 不调用 LLM 做摘要 | `[已定]`（N 待定） |
| D-60 | 一轮只允许一个活跃 run | 避免会话内状态交错 | `[已定]` |
| D-61 | 投影确定性 + `ProjectionRecord` 审计披露 | 使披露图可作研究指标 | `[已定]` |
| D-62 | 级联检测三指标（Round-0 独立共识度 / 披露-改变率 / 扰动测试） | 使「避免级联」的主张可证伪 | `[已定]`（论文阶段实现） |
| D-63 | RAG 管道用 LlamaIndex，但 LLM 与向量模型都写成**适配器**（`CustomLLM` / `BaseEmbedding`）而非换用其官方实现 | 保住自研 LLM 的缓存、用量与异常翻译；key 不进第三方库，业务层仍拿不到明文 | `[已定]` |
| D-64 | 图库存**领域 schema**（`CHANGE_ORDER`/`COMPONENT`/…），不用 LlamaIndex 的 `PropertyGraphIndex` 原生 schema | 既有 Cypher 与「部门→专业」推导一行不改；`assess_grounding` 的 `source="graph"` 语义继续成立 | `[已定]` |
| D-65 | 领域三元组默认用**确定性规则**抽取，LLM 抽取（`SchemaLLMPathExtractor`）为可选项 | 语料字段本就有显式标签，LLM 重抽只换来不确定性与成本（P7）；两者输出同一套 `kg_nodes`/`kg_relations`，可互换 | `[已定]` |
| D-66 | 写库前对每条三元组做 schema 校验（关系名 + 端点类型 + 合法组合），不合法即丢弃并计数 | 下游的专家激活完全建立在图上，一条编造的边会推错整个专业集；且写入用静态 Cypher 而非 APOC 动态 label，消除注入面 | `[已定]` |
| D-67 | 向量检索命中一律记为 `source="text"`，仅当 `VECTOR_MIN_SCORE` 开启且达标（或图腿确认）才计为 `graph` | 向量检索**总是**返回 top-k，仅凭此宣称有历史依据会把 `knowledge_based` 抬成 `history_backed`；此类命中以 `vector_only_hits` 警告显式可见 | `[已定]` |
| D-68 | 检索后端降级必须带原因（打印 + 事件日志）；显式指定后端不可用时**直接报错** | `auto` 的静默回退会让「检索不到」伪装成「本来就没有历史案例」，直接污染保证等级（P5） | `[已定]` |
| D-69 | 语料**一单一节点**，不做切分；node id 用 `文件名:单号` 的 UUIDv5 | 118 单平均约 184 字，切分会让「一条证据」与「一张变更单」不再一一对应，破坏可回溯；uuid5 保证重复摄取幂等（Qdrant 只接受 uint64/UUID） | `[已定]` |

---

## 11. 待决事项

| # | 事项 | 备选 | 建议 |
| --- | --- | --- | --- |
| Q-01 | `disciplines` 推导时机 | (a) 检索时实时推导 / (b) 建图时预先打标 | **已定 (b)**：`corpus.py` 解析时即按「部门→专业」静态表打标，写入 Qdrant payload 并建索引，检索期零成本。表已补全语料里出现过的 10 个部门（`rag.py::DEPT_TO_DISCIPLINE`） |
| Q-02 | 各专家默认自主度 | L0 / L1 / L2 | **默认 L1**，高不确定专家升 L2 |
| Q-03 | 工具白名单首批内容 | `search_evidence` / `request_evidence` / `query_subgraph` / `find_similar_cases` / `lookup_standard` | 先实现检索类，规范类后置 |
| Q-04 | 检索配额 | 单专家 N 次 / 全局 M 次 | 单人 2 / 全局 12 |
| Q-05 | Pass 1 多路 query 数量上限 | — | 3–5 路 |
| Q-06 | meta_agent 决策方式 | 纯 LLM / 规则+LLM 混合 | **混合**：规则覆盖可复现部分 |
| Q-07 | 并发上限 | — | 需实测本地 Ollama 与 Neo4j 承受能力 |
| Q-08 | 最少有效专家数 | 3 / 其它 | 3 |
| Q-09 | HITL 呈现介质 | 对话 / CLI / 外部系统 | 对话（已定）；实现可先做结构化接口 |
| Q-10 | 直接父结构间接案例 | 保留 / 不保留 | **保留**，成本低收益高 |
| Q-11 | OTel span 粒度 | 阶段级 / LLM 调用级 | 两者都要，见 §9.3 |
| Q-12 | 提交规范是否强制 | 约定 / pre-commit + CI | 建议强制 |
| Q-13 | 代码规范工具链 | ruff + pytest | 沿用 `CDIACR` 配置 |
| Q-14 | 会话滑动窗口 N | 3 / 5 / 其它 | 3（demo） |
| Q-15 | `own_previous` 是否含完整 `rationale` | 含 / 不含 | **含**——只看结论无法判断推理是否仍成立 |
| Q-16 | 是否回传自己的 CoT（`reasoning_content`） | 回传 / 不回传 | **不回传**——省 token，结构化意见已含理由 |
| Q-17 | 振荡检测阈值 | 连续 2 轮反向 / 其它 | 连续 2 轮转 `manual_review`；demo 先只记录不拦截 |
| Q-18 | 动态披露策略 `f(Task, Conflict, Consensus)` | — | v2；demo 用固定两档 |
| Q-19 | `Claim.condition` 是否必填 | 必填 / 选填 | 选填，但强建议填写（条件丢失是本设计的主要风险） |

---

## 12. 尚未讨论的模块

| # | 主题 | 说明 |
| --- | --- | --- |
| 1 | 状态与归约器规格 | 每个状态字段的 reducer、结合律测试 |
| 2 | 并发与取消语义细节 | 并发上限、Ollama 连接池、本地模型并行承载能力 |
| 3 | 可观测性 span 结构 | 具体 span 层级与属性命名 |
| 4 | `rag/` 端口契约签名 | **已定**：端口保持 async，同步的 LlamaIndex / neo4j / qdrant 调用一律 `asyncio.to_thread` 下沉到工作线程；异常在适配层翻译一次。见 [`rag.md`](rag.md) |
| 5 | 测试策略 | 契约测试 / 假端口 / fixture 回归 / testcontainers 集成 |
| 6 | 配置与版本矩阵 | **已定**：pydantic v2；`llama-index-core>=0.14,<0.15`、`llama-index-vector-stores-qdrant>=0.10,<0.11`、`qdrant-client>=1.17,<1.18`（minor 与服务端镜像差不得超过 1）；embedding 维度变更必须重建集合 |
| 7 | 代码规范细则 | 命名、分层、docstring 语言、禁止事项清单 |
| 8 | 提交准则 | Conventional Commits、分支、PR、pre-commit |
| 9 | 安全与密钥管理 | `.env` 约定、密钥扫描、日志脱敏 |
| 10 | Prompt 与模型版本管理 | 论文可复现性要求 |
| 11 | 权限与写入通道 | 图谱提案 → 批准 → 落库的完整流程 |

---

## 13. 相关仓库（参考实现与素材）

| 仓库 | 角色 |
| --- | --- |
| `D:\workspace\变更方案生成` | 旧实现（Dify DSL 56 节点 + FastAPI GraphRAG 服务 + 3 个脚本） |
| `D:\workspace\ec` | 旧实现的整理版（含 README、论文材料） |
| `D:\workspace\CDIACR` | 类型化重写版（`change_assistant`：langgraph + pydantic v2 + spec 驱动 + AGENTS.md 协作规则）；本设计的多处反例与参照来源 |

---

*本文件为讨论中的设计基线，随讨论更新。修改时请同时更新 §10 决策记录与 §11 待决事项的状态。*
