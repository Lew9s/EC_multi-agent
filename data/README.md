# data/ —— 语料目录

## 本目录**不含**任何语料

真实语料（船厂工程变更单）属于业务数据，**刻意不纳入版本库**：

* 本仓库是 public，一旦推送即无法真正回收；
* 语料与代码的生命周期、许可、脱敏要求都不同，混在一起会互相绑架。

因此 `.gitignore` 里写的是 `data/*` + `!data/README.md`：
目录结构保留，数据文件不进来。

## 自己准备语料

把符合下面格式的文件放到本目录（默认文件名 `zahuo.txt`），即可跑通摄取与全部测试：

```bash
# 默认路径 data/zahuo.txt
python -m ec_renew.ingest

# 或者用环境变量指到别处（.env）
DATA_DIR=data
CORPUS_FILE=zahuo.txt
```

## 文件格式

纯文本（UTF-8），**变更单之间用一行 `!@#$%^&*` 分隔**，每张变更单由若干
`字段名:值` 行组成，字段值可以续行（`变更内容` 常常是多行）：

```
单号:H-02-1
变更原因:设计公司修改
变更时间点:施工前修改
变更内容:1.因设计公司修改污水井位置 301分段FR36污水井处更换加厚板 结构修改如下:
变更对象:污水井
签收部门:定额室，质保部，生产部
!@#$%^&*
单号:H-02-2
...
```

| 字段 | 是否必需 | 去向 |
| --- | --- | --- |
| `单号` | 建议 | 图节点 `CHANGE_ORDER.name`；缺失时自动生成 `AUTO-00N` 并记入 `missing` |
| `变更原因` | 建议 | `REASON` 节点 + `HAS_REASON` |
| `变更时间点` | 建议 | `TIME_POINT` 节点 + `OCCURS_AT` |
| `变更对象` | 建议 | `COMPONENT` 节点 + `MODIFIES` |
| `签收部门` | 建议 | `DEPARTMENT` 节点 + `SIGNED_BY`；并据此推出 `disciplines`（E01–E06） |
| `变更内容` | 建议 | 补抽结构引用（`301分段`、`FR36`），并生成 `PART_OF` 边 |

* 分隔符是**正则**，可用 `CORPUS_SEPARATOR` 覆盖。
* 部门名到专业的映射是静态表，见 `src/ec_renew/rag.py::DEPT_TO_DISCIPLINE`。
  出现表外的部门会落到默认专业 `E03`（任何变更都涉及规范/质量）——
  换语料时记得同步这张表。
* 若你的语料是**没有字段标签**的自由文本，把抽取器换成 LlamaIndex 的
  `SchemaLLMPathExtractor`（`KG_EXTRACTOR=llm` 或 `--extractor llm`），
  详见 [`docs/rag.md`](../docs/rag.md) §4.2。

## 没有语料时会怎样

| | 行为 |
| --- | --- |
| `python -m ec_renew.ingest` | 立刻报错并提示路径，不会产出空索引 |
| `python -m ec_renew.cli --offline` | **正常可用**（走内存 fixture，不读语料） |
| `pytest` | 依赖真实语料的用例自动 skip；依赖合成语料的用例**照常运行** |

仓库自带一份合成语料 `tests/fixtures/sample_change_orders.txt`（8 张虚构变更单，
格式完全一致），CI 靠它验证「摄取 → 建图 → 检索」整条链路，
因此**不需要业务数据也能确认管道是通的**。
