# 技能：工程变更方案撰写（`plan_writing`）

## 能力

把一次已收束的评审，写成一份**可施工/可采购**的变更方案：

* **骨架由契约固定**：章节是枚举（`PlanHeading`），模型不能自创标题；
* **正文自由**：每节的 `body` 由模型撰写，不受模板腔束缚；
* **需要可回溯的部分结构化**：`claims[] = {text, evidence_ids}`，每条都要引用本轮证据；
* **不带证据的话有唯一出口**：`assumption_notes`（写在这里就会被看见，混进 claims 会被守卫剔除）。

## 输入 `PlanContext`

| 字段 | 说明 |
| --- | --- |
| `request` | 归一化后的变更请求 |
| `opinions` | **完整意见**（含专家身份、决策、约束、不确定项、风险等级、`basis`、`evidence_ids`） |
| `evidence` | 本轮冻结基线的证据（`evidence_id` + 摘要）——**可引用清单** |
| `consensus_status` / `review_reason` / `conditions` | 共识终态；`conditions` 必须全部落进方案 |
| `human_facts` | 人类提供的事实（原文，`source="human"`） |
| `session_notes` | 会话上下文（历轮摘要、用户约束） |
| `previous_plan` / `plan_feedback` | §5.9 迭代回路：上一版方案 + 用户逐条意见 |

## 为什么方案撰写者看得到**专家身份**（D-100）

`ExpertOpinion` 在**评审者之间**是匿名披露的（D-77）：防的是同行压力污染独立判断。
方案撰写者不是评审者，是**汇总者**——它需要身份做专业归口（「这条由哪个专业负责」），
也需要知道「谁明确反对」才能把异议写进待确认事项。匿名范围按**角色**划，不是全局规则。

## 输出 `PlanDraft`

```json
{
  "sections": [
    {"heading": "scope", "body": "…叙述…",
     "claims": [{"text": "更换 301 分段 FR36 污水井加厚板", "evidence_ids": ["E-…"]}]}
  ],
  "assumption_notes": ["板厚按原图纸 12mm 计，未获确认"]
}
```

## 守卫（外环，`workflow.plan_guard`）

1. 每条 `claims[].evidence_ids` 必须 ⊆ 本轮 registry，越界即剔除该条（与 D-90 同规则）；
2. `REQUIRED_PLAN_HEADINGS` 里有章节**一条有效 claim 都没有** → 拒绝整份方案，
   状态**降为 `manual_review`**、`review_reason="unsupported_plan"`（§5.7 硬约束）；
3. `conditions` 少一条就补一条（从写出该条件的专家意见取 `evidence_ids`），并告警——
   条件不能丢，否则「有条件通过」在交付物里变成「无条件通过」（D-95）；
4. 失败 / 无 LLM → 回落到确定性模板（`plan_source="template"`），**降级必须显式**（P5）。
