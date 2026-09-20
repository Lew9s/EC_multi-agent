"""Q-06 的「三明治」第二层（D-102）：LLM 只补差集，合并与降级都是显式的。

这个文件钉住五件事：

1. 补差集**只增不减**，并且真的到达派发——否则它只是提示词里的一句话；
2. **越界即退回契约**：编造的 `evidence_id` 不会被静默接受；
3. 证据子集裁剪**真的裁剪投影**，且落事件（§5.4.3 的 L3 层：裁剪决定共识基线）；
4. 决策**只跑一次**（首轮之前），`rule` 模式下根本不发起调用；
5. 失败**显式降级**回纯骨架，不是静默继续（P5）。
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from ec_renew.agents.memory import EvidenceRegistry
from ec_renew.agents.meta import RuleSkeleton, merge_decision, parse_decision
from ec_renew.agents.runtime import DECISION_NODE
from ec_renew.contracts import (
    EXPERT_IDS,
    ActivationPlan,
    LLMCallMeta,
    LLMResult,
    MetaDecision,
    RunInput,
    Usage,
)
from ec_renew.errors import ContractViolation
from ec_renew.observability import NullEventLog
from ec_renew.ports import RunContext
from ec_renew.rag import InMemoryRetriever
from ec_renew.workflow import run

REQUEST = "301分段FR36污水井更换加厚板，涉及焊接"
_EID_RE = re.compile(r"E-[0-9a-f]{12}")


class _Sink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def payload(self, name: str) -> dict:
        return next(fields for event, fields in self.events if event == name)


class _ScriptedLLM:
    """按 purpose 分派：决策层回一份脚本化的差集，其余回一份合法意见。"""

    def __init__(
        self,
        *,
        payload: dict | None = None,
        raw_decision: str | None = None,
        scope_first_for: str | None = None,
    ) -> None:
        self.payload = payload or {}
        self.raw_decision = raw_decision
        self.scope_first_for = scope_first_for
        self.decision_calls = 0
        self.prompts: list[tuple[str, str, str]] = []

    async def complete(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        meta: LLMCallMeta | None = None,
    ) -> LLMResult:
        self.prompts.append((purpose, meta.expert if meta is not None else "", user))
        if purpose == "meta_decision":
            self.decision_calls += 1
            if self.raw_decision is not None:
                content = self.raw_decision
            else:
                body = dict(self.payload)
                if self.scope_first_for:
                    ids = sorted(set(_EID_RE.findall(user)))
                    body.setdefault("evidence_scope", {})[self.scope_first_for] = ids[:1]
                content = json.dumps(
                    {
                        "add_experts": [],
                        "weights": {},
                        "evidence_scope": {},
                        "rationale": "scripted",
                        **body,
                    },
                    ensure_ascii=False,
                )
            return LLMResult(content=content, model="scripted", usage=Usage(calls=1))

        cited = sorted(set(_EID_RE.findall(user))) or ["E-9c4b1e7a2f03"]
        return LLMResult(
            content=json.dumps(
                {
                    "decision": "approve",
                    "rationale": "scripted",
                    "evidence_ids": cited,
                    "risk_level": "low",
                },
                ensure_ascii=False,
            ),
            model="scripted",
            usage=Usage(calls=1),
        )

    def expert_prompts(self, expert: str) -> list[str]:
        return [
            user for purpose, who, user in self.prompts if purpose == "expert" and who == expert
        ]


def _run(llm: _ScriptedLLM, *, meta_decision: str = "llm", sink: _Sink | None = None):
    ctx = RunContext(
        run_id="decision",
        llm=llm,  # type: ignore[arg-type]
        registry=EvidenceRegistry(),
        events=sink or NullEventLog(),
        retriever=InMemoryRetriever(),
    )
    return asyncio.run(run(RunInput(request=REQUEST), ctx, meta_decision=meta_decision))


# --------------------------------------------------------------------------- #
# 1) 只增不减：合并是代码的事，不是提示词里的一句请求
# --------------------------------------------------------------------------- #


def test_merge_keeps_every_skeleton_expert_and_appends_the_difference() -> None:
    skeleton = ActivationPlan(
        active_experts=["E01", "E03", "E06"],
        weights={"E01": 1 / 3, "E03": 1 / 3, "E06": 1 / 3},
        evidence_scope={e: ["E-a", "E-b"] for e in ("E01", "E03", "E06")},
        rationale="规则骨架",
    )
    merged, corrections = merge_decision(
        skeleton,
        MetaDecision(add_experts=["E04", "E02"], rationale="涉及电气与舾装"),
        baseline_ids=["E-a", "E-b"],
    )

    assert merged.active_experts == ["E01", "E02", "E03", "E04", "E06"]
    assert set(skeleton.active_experts) <= set(merged.active_experts)
    assert abs(sum(merged.weights.values()) - 1.0) < 1e-9
    assert any("补差集" in item for item in corrections)


def test_merge_ignores_a_weight_for_an_expert_that_is_not_active() -> None:
    """给未激活专家配权重是模型自己的错，合并层记账但不采纳。"""
    skeleton = ActivationPlan(
        active_experts=["E01", "E03", "E06"],
        weights={"E01": 1 / 3, "E03": 1 / 3, "E06": 1 / 3},
        evidence_scope={e: ["E-a"] for e in ("E01", "E03", "E06")},
    )
    merged, corrections = merge_decision(
        skeleton, MetaDecision(weights={"E05": 9.0}), baseline_ids=["E-a"]
    )

    assert merged.active_experts == ["E01", "E03", "E06"]
    assert "E05" not in merged.weights
    assert any("忽略非激活专家的权重" in item for item in corrections)


def test_a_none_decision_returns_the_skeleton_untouched() -> None:
    skeleton = ActivationPlan(active_experts=["E01"], weights={"E01": 1.0})
    merged, corrections = merge_decision(skeleton, None, baseline_ids=["E-a"])
    assert merged is skeleton
    assert corrections == ()


def test_the_decision_layer_reaches_the_dispatch() -> None:
    """补差集必须真的改变激活集——只写在提示词里不算接线。"""
    sink = _Sink()
    result = _run(_ScriptedLLM(payload={"add_experts": ["E04"]}), sink=sink)

    assert "E04" in result.active_experts
    merged = sink.payload("meta_decision_merged")
    assert "E04" in merged["active"]
    assert abs(sum(merged["weights"].values()) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# 2) 越界即退回契约（§5.4：元智能体不得编造 evidence_id）
# --------------------------------------------------------------------------- #


def test_an_invented_evidence_id_is_a_contract_violation() -> None:
    raw = json.dumps({"evidence_scope": {"E01": ["E-ffffffffffff"]}})
    with pytest.raises(ContractViolation):
        parse_decision(raw, allowed_evidence=["E-9c4b1e7a2f03"])


def test_an_unknown_expert_id_is_a_contract_violation() -> None:
    with pytest.raises(ContractViolation):
        parse_decision(json.dumps({"add_experts": ["E99"]}), allowed_evidence=[])


def test_a_negative_weight_is_a_contract_violation() -> None:
    with pytest.raises(ContractViolation):
        parse_decision(json.dumps({"weights": {"E01": -1}}), allowed_evidence=[])


def test_an_unusable_decision_degrades_to_the_pure_skeleton() -> None:
    """重试耗尽 → 回到骨架，并且**说出来**（warning + degradation 事件）。"""
    sink = _Sink()
    llm = _ScriptedLLM(payload={"add_experts": ["E04"]}, raw_decision="这不是 JSON")
    result = _run(llm, sink=sink)

    assert "meta_decision_failed" in result.warnings
    assert "degradation" in sink.names()
    assert sink.payload("degradation")["level"] == "rule_skeleton"
    # 降级后激活集完全由骨架决定：越界的那次提案没有生效。
    assert "E04" not in result.active_experts
    assert llm.decision_calls == 2, "契约失败必须先带反馈重试一次"


# --------------------------------------------------------------------------- #
# 3) 证据子集裁剪真的裁剪投影，并且落事件
# --------------------------------------------------------------------------- #


def test_a_trimmed_evidence_scope_is_applied_and_logged() -> None:
    sink = _Sink()
    llm = _ScriptedLLM(scope_first_for="E01")
    _run(llm, sink=sink)

    decision_prompt = next(user for purpose, _, user in llm.prompts if purpose == "meta_decision")
    shown = sorted(set(_EID_RE.findall(decision_prompt)))
    assert len(shown) >= 2, "前提：本轮可引用证据不止一条，裁剪才有观察价值"

    for prompt in llm.expert_prompts("E01"):
        assert shown[0] in prompt
        assert all(other not in prompt for other in shown[1:])

    assert "evidence_scope_trimmed" in sink.names()
    assert sink.payload("evidence_scope_trimmed")["kept"]["E01"] == 1


def test_an_empty_scope_falls_back_to_the_whole_baseline() -> None:
    """给专家「零证据」不是裁剪，是让他没法判断——空子集回落整份基线。"""
    skeleton = ActivationPlan(
        active_experts=["E01", "E03", "E06"],
        weights={"E01": 1 / 3, "E03": 1 / 3, "E06": 1 / 3},
    )
    merged, _ = merge_decision(
        skeleton, MetaDecision(evidence_scope={"E01": []}), baseline_ids=["E-a", "E-b"]
    )
    assert merged.evidence_scope["E01"] == ["E-a", "E-b"]


# --------------------------------------------------------------------------- #
# 4) 只跑一次；rule 模式下不发起调用
# --------------------------------------------------------------------------- #


def test_the_decision_runs_once_before_the_first_round() -> None:
    llm = _ScriptedLLM()
    _run(llm)
    assert llm.decision_calls == 1
    assert llm.prompts[0][0] == "meta_decision", "决策必须发生在第一次 think 之前"


def test_rule_mode_never_calls_the_decision_layer() -> None:
    sink = _Sink()
    llm = _ScriptedLLM(payload={"add_experts": ["E04"]})
    result = _run(llm, meta_decision="rule", sink=sink)

    assert llm.decision_calls == 0
    assert "meta_decision_skipped" in sink.names()
    assert "E04" not in result.active_experts, "关掉决策层后激活集必须完全由骨架决定"


# --------------------------------------------------------------------------- #
# 5) 可复现：同输入同激活集（P7）
# --------------------------------------------------------------------------- #


def test_the_same_decision_input_produces_the_same_activation() -> None:
    payload = {"add_experts": ["E04"], "weights": {"E04": 3.0}}
    first_sink, second_sink = _Sink(), _Sink()
    first = _run(_ScriptedLLM(payload=payload), sink=first_sink)
    second = _run(_ScriptedLLM(payload=payload), sink=second_sink)

    assert first.active_experts == second.active_experts
    assert first_sink.payload("meta_decision_merged")["weights"] == (
        second_sink.payload("meta_decision_merged")["weights"]
    )


def test_the_decision_step_is_recorded_before_it_runs() -> None:
    """§9.4：决策也是一次 LLM 调用，必须有成对的 step / step_done。"""
    sink = _Sink()
    _run(_ScriptedLLM(), sink=sink)

    steps = [fields for name, fields in sink.events if name == "step"]
    decision_steps = [fields for fields in steps if fields.get("node") == DECISION_NODE]
    assert len(decision_steps) == 1
    assert decision_steps[0]["kind"] == "llm"
    assert any(
        name == "step_done" and fields.get("node") == DECISION_NODE
        for name, fields in sink.events
    )


def test_the_skeleton_still_produces_a_consumer_ready_plan() -> None:
    """骨架的既有契约不变：它仍然是 ActivationPlan 的生产者（§5.4.8）。"""
    plan = RuleSkeleton().plan(request=REQUEST, baseline_ids=["E-a"])
    assert set(plan.active_experts) <= set(EXPERT_IDS)
    assert plan.evidence_scope
