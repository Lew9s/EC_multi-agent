# RAG 管道（LlamaIndex）

> **状态**：已实现并离线 / 在线双路径验证
> **范围**：摄取、存储 schema、检索、与 workflow 的接缝、降级策略、运维
> **上游**：[`design.md`](design.md) §5.2（RAG 双接口）、D-11 ~ D-18

---

## 1. 一张图

> 本文件里的裸文件名（`corpus.py` / `retriever.py` / `graph_store.py` …）都位于
> `src/ec_renew/rag/`；领域图与 Cypher 在 `rag/graph.py`（原 `rag.py`）。

```
DATA_DIR/CORPUS_FILE 指向的变更单语料（!@#$%^&* 分隔；**不随仓库分发**，见 data/README.md）
        │
        ▼  corpus.py：一单一 Document，字段结构化 + disciplines 打标
   ┌──────────────────────────── IngestionPipeline ────────────────────────────┐
   │                                                                            │
   │  transformations=[DomainTripletExtractor, ZhipuEmbedding]                  │
   │        │                                      │                            │
   │        ▼ kg_nodes / kg_relations              ▼ 向量                        │
   └────────┼──────────────────────────────────────┼────────────────────────────┘
            ▼                                      ▼
   graph_store.py → Neo4j                  vector_store.py → Qdrant
   （领域 schema，静态 Cypher）              （COSINE，维度=EMBEDDING_DIMENSIONS）
            │                                      │
            └──────────────┬───────────────────────┘
                           ▼
              retriever.py：LlamaIndexRetriever
              ├─ link()     图：请求文本 → 图中实体
              ├─ expand()   图：实体 → 父结构 / 历史部门 → 专业（E01–E06）
              └─ prefetch() 向量腿 + 图腿 → RRF 融合 → EvidenceBundle
                           ▼
              workflow.py：assess_grounding → 选专家 → Delphi 两轮
```

---

## 2. 选型

| 项 | 选择 | 落点 |
| --- | --- | --- |
| 编排 | LlamaIndex `IngestionPipeline` + `VectorStoreIndex` | `ingest.py` / `retriever.py` |
| LLM | DeepSeek `deepseek-flash`（项目原有 `DeepSeekLLM`，httpx 直连） | `llm.py`（桥接） |
| 向量模型 | 智谱 `embedding-3`（OpenAI 兼容协议） | `embeddings.py` |
| 向量库 | Qdrant（docker compose） | `vector_store.py` |
| 图库 | Neo4j（docker compose） | `graph_store.py` |
| 融合 | Reciprocal Rank Fusion | `retriever.py` |

### 2.1 为什么 LLM 要「桥接」而不是换成 LlamaIndex 的

项目原有的 `DeepSeekLLM` 已经带了响应缓存、用量统计和异常翻译，key 也只在
`config.py` 与它内部存在。`LlamaLLMBridge`（`CustomLLM`）把 `LLMPort` 包一层，
于是 LlamaIndex 内部的每次调用依然走同一套缓存与计费口径，
**LlamaIndex 拿不到 key**，也不需要引入 `openai` SDK。

### 2.2 为什么要自己写 embedding 适配器

和上面同理：`ZhipuEmbedding` 是 `BaseEmbedding` 的实现，明文 key 只向
`config` 取一次，存进 pydantic 的 `PrivateAttr`（不进 `repr` / `model_dump` /
事件日志），再由 httpx client 的请求头持有。测试 `test_zhipu_api_key_never_leaks`
把这条钉死。

---

## 3. 密钥管理

* **唯一来源**：`.env`（已被 `.gitignore` 忽略）。模板见 `.env.example`。
* **唯一读取点**：`config.py`。所有 key 一律是 `SecretStr`。
* **业务层拿不到明文**：`workflow` / `experts` / `rag` 只见 `Settings` 对象。
* **不落日志**：事件日志只写模型名、用量、集合名，不写 prompt 原文、
  连接串和密钥（design.md §9.3）。
* 需要两个 key：`DEEPSEEK_API_KEY`（LLM）与 `ZHIPU_API_KEY`（向量）。
  **缺 key 时不会静默降级**，而是启动即报错 —— 见 §7。

---

## 4. Neo4j：领域 schema

保留 `design.md` §0.3 的原始 schema，`rag/graph.py` 里既有的 Cypher 一行没改：

```
(:CHANGE_ORDER {name, group_key})   name 即单号，如 H-02-1
(:COMPONENT   {name})
(:DEPARTMENT  {name})
(:REASON      {name})
(:TIME_POINT  {name})

(:CHANGE_ORDER)-[:MODIFIES]->(:COMPONENT)
(:CHANGE_ORDER)-[:SIGNED_BY]->(:DEPARTMENT)
(:CHANGE_ORDER)-[:HAS_REASON]->(:REASON)
(:CHANGE_ORDER)-[:OCCURS_AT]->(:TIME_POINT)
(:COMPONENT)-[:PART_OF]->(:COMPONENT)          ← 父结构（分段）
```

实测规模（实验室语料 118 张变更单；语料属业务数据，不进版本库）：

| | 数量 |
| --- | --- |
| `CHANGE_ORDER` | 118 |
| `COMPONENT` | 119 |
| `DEPARTMENT` / `REASON` / `TIME_POINT` | 10 / 39 / 8 |
| `SIGNED_BY` / `MODIFIES` / `HAS_REASON` / `OCCURS_AT` / `PART_OF` | 728 / 196 / 118 / 118 / 43 |

其中 `SIGNED_BY=728`、`HAS_REASON=118`、`OCCURS_AT=118` 与历史遗留图谱的
计数**完全一致**，可作为解析未丢字段的交叉验证。

### 4.1 写入为什么用静态 Cypher

不用 `apoc.merge.node`：那需要把 label 拼进查询串，既是注入面，也让
「合法 schema」失去编译期约束。领域关系只有 5 种、端点组合固定，
写死成 5 条 `UNWIND + MERGE`（`graph_store.py`）反而更安全、更快，
部署端也不必装 APOC 插件。

任何一条三元组在写库前都要过 `validate_triple()`：关系名、端点类型、
`KG_VALIDATION_SCHEMA` 三重校验，不合法的边直接丢弃并计数。
宁可少一条编造的边，也不能让它污染 `expand()` 推出的专业集 ——
下游的专家激活完全建立在这个图上。

### 4.2 抽取为什么默认用规则

本语料的字段本来就带显式标签（`单号/变更原因/变更时间点/变更对象/签收部门`），
用 LLM 把已有标签再猜一遍只会换来不确定性与 token 账单（design.md P7）。
真正需要语义理解的只有「变更内容」里的结构引用（`301分段`、`FR36`），
那部分用保守正则补抽，规则简单且可解释。

需要处理**没有字段标签**的自由文本时，把抽取器换成 LlamaIndex 的
`SchemaLLMPathExtractor` 即可（`--extractor llm`）：输出沿用同一套
`KG_NODES_KEY` / `KG_RELATIONS_KEY` 元数据键，下游写库代码一行不用改。

---

## 5. Qdrant：向量库

| 项 | 值 |
| --- | --- |
| 集合 | `QDRANT_COLLECTION`（默认 `ec_renew_change_orders`） |
| 距离 | COSINE |
| 维度 | `EMBEDDING_DIMENSIONS`（默认 2048，`embedding-3` 支持 2048/1024/512/256） |
| 粒度 | **一单一节点**，不再切分 |
| payload 索引 | `case_id` / `group_key` / `disciplines` / `component` |

两个刻意的决定：

1. **集合由本项目显式创建**，不让 `QdrantVectorStore` 在首次写入时隐式建。
   维度不一致时检索会**静默地什么都查不到**，所以宁可在启动阶段就报错。
2. **不切分**。118 张变更单平均约 184 字，远小于 embedding-3 的窗口，
   切分只会让「一条证据」和「一张变更单」不再一一对应，破坏可回溯性。
   语料变大时在 `IngestionPipeline` 的 `transformations` 前面加一个
   `SentenceSplitter` 即可。

node id 是 `文件名:单号` 的 **UUIDv5**（`corpus.py::document_id`）：
* 用 UUID 是因为 Qdrant 的 point id 只接受 uint64 或 UUID；
* 用 uuid5 而不是 uuid4 是因为**重复摄取必须落在同一个点上**（幂等）。

---

## 6. 检索

`LlamaIndexRetriever` 实现 `ports.RetrieverPort`，三个方法各司其职。

### `link()` / `expand()`

直接复用 `rag/graph.py` 的 Cypher（Cypher 只允许出现在 `rag/graph.py` 与 `rag/graph_store.py`）。
`link()` 在 CONTAINS 命中后多过一道 `is_partial_identifier()`：请求里写着
`FR36` 时，图里的 `FR3` 也会被 CONTAINS 命中，否则意图补全会凭空多出一个
不存在的实体，`expand()` 再据此推出错误的专业集。

### `prefetch()` —— 外环的共享事实基线

```
queries ─┬─ 向量腿：每条 query 取 top-k，按 1/(k+rank) 计分
         └─ 图腿：terms 命中数排序
                    │
                    ▼ RRF 融合（只用排名，不用分数）
              按融合分排序 → 截断到 top_k → EvidenceBundle
```

**为什么融合用 RRF 而不是分数加权**：向量余弦与图匹配计数不在同一个量纲上，
归一化等于悄悄引入一个说不清来历的系数；RRF 只用排名，无需调参，
且对两条腿的分数尺度不敏感。

### 6.1 `source` 的诚实性（关系到保证等级）

`assess_grounding()` 只把 `source == "graph"` 的条目算作「有历史依据」。
于是：

| 情况 | `source` | 是否计入 history |
| --- | --- | --- |
| 图腿命中（结构化/词面匹配） | `graph` | ✅ |
| 仅向量命中，且 `VECTOR_MIN_SCORE` 已开启且分数达标 | `graph` | ✅ |
| 仅向量命中（默认） | `text` | ❌ |

**默认不算**，因为向量检索**总是**返回 top-k，哪怕一条都不相关。仅凭
「最近的 k 条」就宣称有历史依据，会直接把 `knowledge_based` 抬成
`history_backed`。这类命中不会消失，而是：既作为 `text` 证据进入基线，
又通过 `warnings` 里的 `vector_only_hits:<n>` **显式可见**（P5：降级必须显式）。

要把高相似度命中也算作历史依据，把 `VECTOR_MIN_SCORE` 设为正数。

**已实测标定**（本语料 118 单 + 智谱 `embedding-3`，36 条标注 query：16 条域内改写 + 20 条域外）：

| | top1 相似度 |
| --- | --- |
| 域内 | 最低 0.3551，p25 0.3818，中位 0.4164，最高 0.4741 |
| 域外 | 最低 0.1082，p75 0.2370，中位 0.2017，最高 0.3390 |

两者之间有一条干净的空隙 **[0.339, 0.355]**，故 `.env` 取 `VECTOR_MIN_SCORE=0.35`（空隙中点）：
域内 16/16 通过，域外 0/20 误放。

> ⚠ **不要凭直觉设 0.5 / 0.8**。`embedding-3` 的余弦分布整体偏压缩
> （本语料约 0.1~0.47），设 0.5 会把**全部**真实命中拒掉，等于白开这个开关。
> 换语料或换向量模型后必须重新标定。复现脚本见 §10。

### 6.2 证据正文 = 语料原文

`EvidenceRegistry` 是内容寻址的（`sha1(source|content)`）。两条腿取到
**同一个字符串**，才会归并成同一条证据、同一个 `evidence_id`。
因此 `_content()` 优先用 `corpus.py` 解析出的**原文**，
图谱里的结构化摘要只在语料文件缺失时兜底。

---

## 7. 降级矩阵

`RAG_BACKEND`（或 `--rag`）：

| 模式 | 组成 | 显式指定但不可用时 |
| --- | --- | --- |
| `llamaindex` | Qdrant + Neo4j（RRF） | **直接报错** |
| `graph` | 仅 Neo4j | **直接报错** |
| `memory` | 内存 fixture | — |
| `auto` | 依次探测，取第一个可用 | 每次回退都记入 `notes` |

`auto` 的降级链**永远带原因**，并由 `cli.py` 打印、写入事件日志
（`retriever_selected`）。绝不出现「悄悄换成内存检索却看起来一切正常」——
那会让「检索不到」伪装成「本来就没有历史案例」。

缺 `ZHIPU_API_KEY` 时 `build_embed_model()` 直接抛错而不是换成假向量；
只有调用方显式 `offline=True` 才用 `FakeEmbedding`。

`FakeEmbedding` 不是随机数，而是字符 unigram+bigram 的 hashing 向量器，
所以离线跑出来的检索结果**与字符重合度相关**，是有意义的，而非噪声。

---

## 8. 运维

```bash
# 1) 起存储
docker compose up -d

# 2) 填 key
cp .env.example .env      # 然后填 DEEPSEEK_API_KEY / ZHIPU_API_KEY

# 3) 自备语料（真实业务数据不进版本库，格式见 data/README.md）
#    读 DATA_DIR / CORPUS_FILE 指向的文件，两者都在 .env 里配置

# 4) 摄取（首次会建集合、建约束、写图与向量）
python -m ec_renew.rag.ingest

# 常用变体
python -m ec_renew.rag.ingest --offline              # 无 key 自检（写入的是假向量！）
python -m ec_renew.rag.ingest --recreate             # 删掉并重建 Qdrant 集合
python -m ec_renew.rag.ingest --reset                # 先清空本项目领域图再写
python -m ec_renew.rag.ingest --graph-only           # 只重建图，不花 embedding 的钱
python -m ec_renew.rag.ingest --json                 # 机器可读结果

# 5) 跑
python -m ec_renew.interface.cli -r "301分段FR36污水井更换加厚板"
python -m ec_renew.interface.cli --rag graph -r "..."      # 只用图，不走向量
python -m ec_renew.interface.cli --offline -r "..."        # 全离线（FakeLLM + 内存检索）
```

> ⚠ **语料不随仓库分发**。它属于业务数据，`.gitignore` 里写的是
> `data/*` + `!data/README.md`。缺失时 `ingest` 会明确报错并提示路径，
> 不会默默产出一个空索引；`pytest` 里依赖真实语料的用例会 skip，
> 而依赖 `tests/fixtures/sample_change_orders.txt`（合成语料）的用例照常运行。

> ⚠ `--offline` 摄取把**假向量**写进 `QDRANT_COLLECTION`。它只用于自检；
> 自检完请用不带 `--offline` 的命令重跑，或指定一个临时集合名。

### 换向量维度

`embedding-3` 支持 2048 / 1024 / 512 / 256。改 `EMBEDDING_DIMENSIONS` 后
**必须** `--recreate`：旧点与新查询向量维度不同，检索会直接失败。
`ensure_collection()` 会主动比对维度并给出明确报错。

---

## 9. 未做 / 已知限制

| 项 | 说明 |
| --- | --- |
| `search()`（agent 工具侧） | design.md D-11 的双接口只实现了 `prefetch()`。当前专家是 L0（单次调用），还没有 L1/L2 的自主补检索，所以 `search()` 尚无调用方。`EvidenceRegistry` 已就位，补上时无需改结构。 |
| 证据请求通道 | `EvidenceRequest` 未实现；轮次基线目前不中途扩张（`memory.py` 里已有注释）。 |
| 增量摄取 | 目前是全量重跑（MERGE 幂等）。语料规模上来后需要按文件哈希跳过未变文档。 |
| 真实 embedding 的阈值校准 | **已完成**：`VECTOR_MIN_SCORE=0.35`，基于 36 条标注 query 实测（见 §6.1）。样本量不大，换语料/换模型后需重标。 |
| 智谱通道的真机联调 | **已完成**：真实 API 联调通过（`embedding-3` 返回 2048 维）。请求形状、批内乱序重排、429/5xx 重试与异常翻译另有 `httpx.MockTransport` 测试覆盖（`test_zhipu_*`）。 |
| `--extractor llm` 的抽取质量 | 接线已验证（能构造出 `SchemaLLMPathExtractor` + `CustomLLM` 桥接），但抽取质量需要真实 DeepSeek key 才能评估。本语料用 `rule` 更准更省，该选项是为**无字段标签的自由文本**准备的。 |
| `PART_OF` 覆盖 | 只在**恰好一个**分段被提及时才建边（歧义宁缺勿错），因此 43 条远少于 196 条 `MODIFIES`。要提覆盖率需要更细的分段识别（图纸号/肋位），不是正则能可靠解决的。 |
| 旧 Qdrant 集合 | 库里另有两个历史遗留集合（`graph_collection`、`change_order_chunks`，1024 维 bge-m3），与本项目的 2048 维不兼容，按约定保留未动。 |

---

## 10. 重新标定阈值

换语料、换向量模型或换 `EMBEDDING_DIMENSIONS` 之后，`VECTOR_MIN_SCORE` 必须重标：

```bash
python -m ec_renew.rag.calibrate            # 用内置的 36 条标注 query
python -m ec_renew.rag.calibrate --json     # 机器可读
python -m ec_renew.rag.calibrate --queries my_labels.json
```

`my_labels.json` 形如 `{"in_domain": [...], "out_domain": [...]}`。

脚本做法：两组 query 各取 top-1 相似度，在细网格上扫描阈值，取
**准确率最高、且离所有观测分数最远**的那个（最大间隔），
也就是落在可分空隙正中、两侧都留余量的取值。

本语料实测输出：

```
域内 n=16  top1: min 0.3551 / median 0.4164 / max 0.4741
域外 n=20  top1: min 0.1082 / median 0.2017 / max 0.3390
可分空隙: 域外最高 0.3390 → 域内最低 0.3551
推荐 VECTOR_MIN_SCORE=0.3472（域内通过 16/16，域外误放 0/20）
```

`.env` 里写的是 `0.35`（该值取两位小数），仍在安全带 `(0.3390, 0.3551]` 内。

> 标定集只有 36 条，是**量级参考**而非统计保证。生产上应扩到数百条，
> 并注意域外样本要覆盖「与船体词汇有汉字重合」的干扰项。
