# 技能：专家评审（`expert_review`）

> 版本 **1.0.0** ｜ 输入 `ExpertTask` ｜ 输出 `ExpertOpinion`
> 实现：`personas.py`（谁）· `prompt.py`（怎么说）· `runner.py`（怎么做）
> 用例：`tests/test_skill_expert_review.py`

## 什么时候用

让**一个 agent** 对某项工程变更，站在**一个专业视角**上，给出一份结构化评审意见。它是内环
子智能体的能力；同一技能被六个专业复用（差别只在 persona），也可被其它 agent 装载。

## 输入 / 输出契约

输入 `ExpertTask`（`contracts.py`）：请求、本轮证据子集（`evidence`，带 `evidence_id`）、
子问题、轮次与模式（`initial` / `revise`）、`revision.own_previous`（只回灌自己的上一轮）、
`cross_agent`（匿名 claim + 硬约束）。

输出 `ExpertOpinion`：`decision`、**`basis`**、`rationale`、`evidence_ids`、`claims`、
`constraints`、`uncertainties`、`risk_level`、`confidence`。

契约由 `runner.parse_opinion()` 校验：`evidence_ids ⊆ 本轮允许集`（越界即契约违规）；
`claim.discipline` 由**实际专家身份**覆盖，不采信模型自报（D-77）。

## 判断准则（本技能的核心）

| 项 | 规则 |
| --- | --- |
| 事实性断言 | **必须**引用本轮 `evidence_id`，不得编造（P3） |
| 专业判断 | **可以**基于领域通识/规范常识/工程经验作出。**本轮证据给不出技术细节时仍要判断**，不要弃权 |
| `basis` | 必须声明：`evidence` / `knowledge` / `mixed`。声明 `knowledge` 会让保证等级标为 `knowledge_based` —— 这是诚实，不是惩罚 |
| `abstain` | **仅**两种情况允许：① 超出本专业范围；② 请求本身无法判定。**「参数没给全」不是弃权理由** |
| 缺失信息 | 写进 `uncertainties`（不确定什么）与 `constraints`（必须先满足什么） |

> **为什么单列这条**：P3（事实须可回溯）管的是**事实性断言**，不约束**专业判断**；把它读成
> 「没有案例证据就不能判断」，会把专业判断一并禁掉。设计本就有 `knowledge_based` 这一档
> （§5.5.1 / §5.5.6 / D-42）：无历史**也要评估**，只是降级标注。

## 失败方式（都必须是显式的）

1. **契约重试**：输出无法解析时，把违规原因回灌再试一次（`max_contract_retries`，默认 1）。
2. **修复**：重试仍失败 → 截断越界字段、标 `partial=True`，落 `contract_repaired` 事件。
3. **弃权**：修复也做不到 → `abstain_opinion()`，`uncertainties` 里带原始错误。
   ⚠ 它与「模型自报弃权」共用 `decision="abstain"`，靠 `abstain_kind` 区分：模型自报的是
   **判断性弃权**（算交付），这里是**执行失败弃权**（算缺席）（D-93）。

## 不该用它做什么

- 不写库、不改 registry、不选专家（那是守卫与外环的职责，D-71 / §5.4.2）；
- 不做跨专家汇总（那是 Delphi 披露与共识计算的事）；
- 不产生自由文本之外的结构外信息：`rationale` 只进报告与审计，**永不进其它 agent 的 prompt**（D-76）。
