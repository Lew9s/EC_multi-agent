"""**D-97**：缺口回填 —— 证据请求真的在下一轮把基线扩出来。

这个文件钉住五件事：

1. **回填发生在下一轮、且在派发之前**：第 1 轮提出的缺口，第 2 轮**所有**专家都看得到新证据
   （不是只有提请求的那位，也不是拖到第 3 轮）——§5.2.4 的「Round N+1 基线上扩，所有人可见」；
2. **逐条记账**：`satisfied_round` / `satisfied_evidence` 是**每条请求各自**的结论，
   「没去查」与「查了、语料里没有」分得开（此前只有一个 run 级警告，两者糊在一起）；
3. **幂等**：同一个缺口每轮都会被重新派生出来（D-94 是确定性的），但只检索一次；
4. **降级不静默**：没有检索端、检索抛可降级异常，都不让 run 崩掉，且落事件（P5）；
5. **后果跟着变**：回填带回图谱证据时，终局的依据等级从 `knowledge_based` 升为
   `history_backed`——§5.7 的等级是交付物上的一句话，不能与实际证据不一致。
"""

from __future__ import annotations

import asyncio
import json
import re

from ec_renew.agents.guard import LoopState
from ec_renew.agents.memory import EvidenceRegistry
from ec_renew.agents.meta import RuleSkeleton
from ec_renew.contracts import (
    EntityRef,
    EvidenceBundle,
    EvidenceMeta,
    ExpertOpinion,
    GraphExpansion,
    LLMCallMeta,
    LLMResult,
    RunInput,
    Usage,
)
from ec_renew.errors import TransientError
from ec_renew.observability import NullEventLog
from ec_renew.ports import RunContext
from ec_renew.workflow import build_queries, run

REQUEST = "301分段FR36污水井更换加厚板，涉及焊接，需确认合规性"
BASE_LABEL = "SRC-BASE-1"
GAP_LABEL = "SRC-GAP-1"
EID = "E-9c4b1e7a2f03"


class _Sink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def names(self) -> list[str]:
        return [name for name, _ in self.events]


class _ScriptedRetriever:
    """基线 query 与缺口 query 返回**不同内容**的检索端。

    ``gap_extra=False`` 模拟「查了但语料里没有」；``fail_gap=True`` 模拟可降级的检索故障；
    ``same_content=True`` 模拟**真实 run 的实测形态**——缺口 query 命中的还是基线里那一条
    （新登记 0 条）；``base_source="standard"`` 让基线**不含图谱证据**（依据等级 = knowledge），
    从而能验证回填把等级抬上去的那条路径。
    """

    def __init__(
        self,
        *,
        gap_extra: bool = True,
        fail_gap: bool = False,
        same_content: bool = False,
        base_source: str = "graph",
    ) -> None:
        self.queries: list[list[str]] = []
        self._gap_extra = gap_extra
        self._fail_gap = fail_gap
        self._same_content = same_content
        self._base_source = base_source

    async def link(self, request: str) -> list[object]:
        return []

    async def expand(self, entities: list[object]) -> GraphExpansion:
        return GraphExpansion()

    async def prefetch(self, queries: list[str], top_k: int) -> EvidenceBundle:
        self.queries.append(list(queries))
        is_gap = any("需补充" in query for query in queries)
        if is_gap and self._fail_gap:
            raise TransientError("检索端超时（用例构造）")
        if is_gap and not self._gap_extra:
            return EvidenceBundle(round=0)
        if is_gap and self._same_content:
            # 内容与基线那条**逐字相同** → content-addressed 之后是同一个 evidence_id。
            return EvidenceBundle(
                round=0,
                items=[
                    EvidenceMeta(
                        evidence_id="SRC-OTHER-LABEL",
                        source="graph",
                        content="历史案例：301分段FR3 结构变更 X-2018-004",
                        disciplines=["结构"],
                        score=0.9,
                    )
                ],
            )
        return EvidenceBundle(
            round=0,
            items=[
                EvidenceMeta(
                    evidence_id=GAP_LABEL if is_gap else BASE_LABEL,
                    source="graph" if is_gap else self._base_source,  # type: ignore[arg-type]
                    content=(
                        "历史案例：301分段FR36 焊接变更 X-2019-001"
                        if is_gap
                        else "历史案例：301分段FR3 结构变更 X-2018-004"
                    ),
                    disciplines=["结构"],
                    score=0.9,
                )
            ],
        )


class _ReviseLLM:
    """恒回 `revise` + 一条「缺少同类案例」的不确定项 —— 缺口因此每轮都被重新派生。"""

    def __init__(self) -> None:
        self.prompts: list[tuple[int, str, str]] = []

    async def complete(
        self, *, purpose: str, system: str, user: str, meta: LLMCallMeta | None = None
    ) -> LLMResult:
        self.prompts.append((meta.round if meta else 0, meta.expert if meta else "", user))
        cited = sorted(set(re.findall(r"E-[0-9a-f]{12}", user)))
        content = json.dumps(
            {
                "decision": "revise",
                "basis": "mixed",
                "rationale": "方向可行，须先核对同类案例的处理结论",
                "evidence_ids": cited,
                "constraints": ["须核对既有同类变更的处理结论"],
                "uncertainties": ["缺少同类案例与历史先例"],
                "risk_level": "medium",
            },
            ensure_ascii=False,
        )
        return LLMResult(content=content, model="revise", usage=Usage(calls=1))


def _run(
    retriever: object,
    llm: object,
    sink: _Sink | None = None,
    *,
    max_rounds: int = 3,
    run_id: str = "refill",
) -> tuple[RunContext, object]:
    ctx = RunContext(
        run_id=run_id,
        llm=llm,  # type: ignore[arg-type]
        registry=EvidenceRegistry(),
        events=sink or NullEventLog(),
        retriever=retriever,  # type: ignore[arg-type]
    )
    result = asyncio.run(run(RunInput(request=REQUEST), ctx, max_rounds=max_rounds))
    return ctx, result


# --------------------------------------------------------------------------- #
# 0) 「补充资料」的最低条件：缺口 query 必须与原 query **不同**
# --------------------------------------------------------------------------- #


def test_a_gap_query_is_a_different_query_from_every_baseline_query() -> None:
    """这是 @用户 提出的判据：**query 不一样**，补充资料的机制才谈得上成立。

    只看文本层（确定性、无需任何后端）：缺口 query 由 D-94 派生，形式是
    `归一化请求｜需补充：<词表命中的词>`；原 query 集由 `build_queries` 产出，形式是
    `请求 / 实体名 / 实体名 历史变更`。

    两条断言：
    1. 缺口 query **不等于**原 query 集里任何一条（逐字），否则回填就是把同一个检索再跑一遍；
    2. 缺口 query **不是**原请求的复制（长度与字符集都变了），否则「补充」只是字面上的。

    这条用例守的是**机制的可成立性**，与当前知识库里有没有内容无关——若将来有人把缺口 query
    「简化」成直接复用请求词，这里会红。
    """
    request = REQUEST
    baseline_queries = build_queries(
        request, [EntityRef(name="FR36", kind="COMPONENT", in_graph=True, graph_key="FR36")]
    )
    state = LoopState()
    state.latest = {
        "E01": ExpertOpinion(
            expert="E01",
            decision="revise",
            evidence_ids=[EID],
            uncertainties=["缺少规范、标准、图集与证书依据"],
        )
    }
    gaps = RuleSkeleton().gaps(state, request=request)

    assert gaps, "前提：这条不确定项必须能派生出缺口请求"
    for gap in gaps:
        assert gap.query not in baseline_queries, "补充资料的 query 与原 query 逐字相同 = 机制空转"
        assert gap.query != request, "补充资料的 query 不能就是原请求"
        assert request in gap.query, "缺口 query 必须带着原始请求（否则检索失去上下文）"
        assert gap.query.startswith(request), "差异来自**追加**的词表词，而不是换了一个问题"
        hit_part = gap.query[len(request) :]
        assert len(hit_part) > 3, f"追加部分形同虚设：{hit_part!r}"


def test_the_baseline_query_set_is_recorded_in_the_events() -> None:
    """事后**可核对**：原 query 集与缺口 query 都必须能从事件里看到。

    此前两者都没有落盘，于是「缺口 query 与原 query 到底一样不一样」这个问题**无法从运行产物
    回答**——只能靠读源码猜。这个可观测性缺口正是把一次 fixture run 误读成「语料不够」的原因之一。
    """
    sink = _Sink()
    _ctx, _result = _run(_ScriptedRetriever(), _ReviseLLM(), sink, max_rounds=2)

    frozen = [fields for name, fields in sink.events if name == "baseline_frozen"]
    refilled = [fields for name, fields in sink.events if name == "evidence_refill_done"]
    assert frozen and frozen[0]["queries"], "原 query 集必须进事件"
    assert refilled and refilled[0]["queries"], "缺口 query 必须进事件"
    assert set(frozen[0]["queries"]).isdisjoint(refilled[0]["queries"]), "两者不得重合"


# --------------------------------------------------------------------------- #
# 1) 回填发生在下一轮，且在派发之前
# --------------------------------------------------------------------------- #


def test_a_gap_is_refilled_into_the_next_round_for_every_expert() -> None:
    """核心用例：第 1 轮提的缺口 → 第 2 轮的**冻结基线**里有它 → 每位专家的 prompt 里都有它。"""
    sink = _Sink()
    retriever = _ScriptedRetriever()
    llm = _ReviseLLM()
    ctx, result = _run(retriever, llm, sink)

    (gap,) = result.evidence_requests
    assert gap.scope == "cases"
    assert gap.effective_round == 2, "请求从下一轮起生效（D-16）"
    assert gap.satisfied_round == 2, "必须在下一轮就回填，而不是拖到最后一轮"
    assert gap.satisfied_evidence

    # 结构性证据：回填的证据进了第 2 轮的冻结基线。
    assert set(gap.satisfied_evidence) <= set(ctx.registry.baseline(2))

    # 端到端证据：第 1 轮的 prompt 里都没有它，第 2 轮**每一位**专家的 prompt 里都有它。
    new_id = gap.satisfied_evidence[0]
    first = [prompt for rnd, _expert, prompt in llm.prompts if rnd == 1]
    second = [prompt for rnd, _expert, prompt in llm.prompts if rnd == 2]
    assert first and second, f"没抓到两轮的 prompt，用例失去意义：{[r for r, _, _ in llm.prompts]}"
    assert all(new_id not in prompt for prompt in first)
    assert all(
        new_id in prompt for prompt in second
    ), "§5.2.4：扩出来的基线对同一轮**所有**专家可见，而不是只有提请求的那位"

    assert "evidence_refill_done" in sink.names()
    assert "已回填" in result.conclusion
    # 没有把「还没查」当成结论：这条请求确实被查过了。
    assert "evidence_request_unsatisfied" not in result.warnings


def test_the_new_evidence_is_named_as_new_in_the_prompt() -> None:
    """回填的意义在于专家**知道哪几条是新的**（`RevisionContext.new_evidence_ids`）。

    这个字段此前恒为空（注释写着「demo: baseline does not expand mid-run」），而且**根本没被
    渲染进 prompt**——于是基线扩张后，专家看到的只是「更大的一堆证据」，无从知道本轮该重新
    检讨什么。现在它既被算出来（两次冻结基线相减），也被写进 prompt。
    """
    retriever = _ScriptedRetriever()
    llm = _ReviseLLM()
    _ctx, result = _run(retriever, llm)

    (gap,) = result.evidence_requests
    new_id = gap.satisfied_evidence[0]

    first = [prompt for rnd, _expert, prompt in llm.prompts if rnd == 1]
    second = [prompt for rnd, _expert, prompt in llm.prompts if rnd == 2]
    assert all("本轮新增证据" not in prompt for prompt in first), "第 1 轮不是修正轮，没有这一节"
    assert all("本轮新增证据" in prompt for prompt in second)
    assert all(new_id in prompt.split("## 本轮新增证据")[1] for prompt in second)


def test_an_unchanged_baseline_tells_the_expert_so() -> None:
    """对照组：没回填到时，prompt 明确写「与上一轮相同」，而不是留空让人猜。"""
    llm = _ReviseLLM()
    _ctx, _result = _run(_ScriptedRetriever(fail_gap=True), llm)

    second = [prompt for rnd, _expert, prompt in llm.prompts if rnd == 2]
    assert second
    assert all("（无：本轮基线与上一轮相同）" in prompt for prompt in second)


# --------------------------------------------------------------------------- #
# 2) 幂等：同一个缺口只查一次
# --------------------------------------------------------------------------- #


def test_a_refilled_gap_is_never_fetched_twice() -> None:
    retriever = _ScriptedRetriever()
    _ctx, result = _run(retriever, _ReviseLLM(), max_rounds=3)

    assert result.rounds == 3, "前提：缺口被重新派生了三轮（D-94 是确定性派生）"
    assert len(result.evidence_requests) == 1, "同一 (scope, query) 不该在结果里出现两次"
    gap_queries = [batch for batch in retriever.queries if any("需补充" in q for q in batch)]
    assert len(gap_queries) == 1, f"缺口被重复检索了：{gap_queries}"


# --------------------------------------------------------------------------- #
# 3) 三种终局互不相同
# --------------------------------------------------------------------------- #


def test_without_a_retriever_the_gap_is_explicitly_unresolved() -> None:
    sink = _Sink()
    ctx, result = _run(None, _ReviseLLM(), sink, max_rounds=2, run_id="no-retriever")

    (gap,) = result.evidence_requests
    assert gap.satisfied_round is None
    assert "evidence_request_unsatisfied" in result.warnings
    assert "evidence_refill_skipped" in sink.names(), "跳过必须落事件（P5）"
    assert "未回填" in result.conclusion
    assert ctx.registry.baseline(2) == ctx.registry.baseline(1), "没有检索端，基线不该变"


def test_a_refill_that_finds_nothing_is_a_conclusion_not_a_failure() -> None:
    """「查了、语料里没有」必须与「没去查」分开——前者是结论，后者是流程缺陷。"""
    sink = _Sink()
    _ctx, result = _run(_ScriptedRetriever(gap_extra=False), _ReviseLLM(), sink)

    (gap,) = result.evidence_requests
    assert gap.satisfied_round == 2
    assert gap.satisfied_hits == 0
    assert gap.satisfied_evidence == []
    assert "evidence_request_no_hit" in result.warnings
    assert "evidence_request_unsatisfied" not in result.warnings
    assert "语料未命中" in result.conclusion


def test_hits_that_were_already_known_are_not_reported_as_refilled() -> None:
    """**真实 run 的实测形态**：缺口 query 命中了 3 条，但全都早就在本轮基线里 → 基线零增量。

    此前这两件事被合成一句「已回填 3 条」（用的是命中数），于是「什么都没变」被写成「补到了
    3 条」——正好说反。现在 hit 与 new 分开记，并且单独报一条警告：**要补的是语料，不是再检索**。
    """
    sink = _Sink()
    _ctx, result = _run(_ScriptedRetriever(same_content=True), _ReviseLLM(), sink)

    (gap,) = result.evidence_requests
    assert gap.satisfied_round == 2
    assert gap.satisfied_hits == 1, "检索确实命中了东西"
    assert gap.satisfied_evidence == [], "但没有一条是新的：基线不该变大"
    assert "evidence_request_known_only" in result.warnings
    assert "evidence_request_no_hit" not in result.warnings
    assert "都已在本轮基线中" in result.conclusion


def test_a_request_raised_in_the_final_round_can_never_be_refilled() -> None:
    """轮次用尽是**如实汇报**的情形，不是静默丢弃：`effective_round` 已无对应轮次。"""
    _ctx, result = _run(_ScriptedRetriever(), _ReviseLLM(), max_rounds=1)

    (gap,) = result.evidence_requests
    assert gap.effective_round == 2
    assert gap.satisfied_round is None
    assert "evidence_request_unsatisfied" in result.warnings
    assert "未回填" in result.conclusion


def test_a_failing_refill_degrades_explicitly_and_then_stalls() -> None:
    """回填失败 → 基线没变 → 第 2 轮就是**不动点** → 如实报 `stalled`（§5.4.4 / D-82）。

    这条顺带说明回填改变了流程走向：拿不到新证据时不该把剩余轮次跑完（那只会产生共识幻觉），
    而该立刻说「再跑也不会有新信息」；缺口没被满足这件事同时由警告说清。
    """
    sink = _Sink()
    _ctx, result = _run(_ScriptedRetriever(fail_gap=True), _ReviseLLM(), sink)

    (gap,) = result.evidence_requests
    assert gap.satisfied_round is None, "失败就不能记成已回填"
    assert "evidence_refill_failed" in sink.names()
    assert "evidence_request_unsatisfied" in result.warnings
    assert result.consensus_status == "stalled"
    assert result.rounds == 2, "没有新信息就不该空跑到轮次上限"


# --------------------------------------------------------------------------- #
# 4) 回填的后果：依据等级与保证等级跟着证据走
# --------------------------------------------------------------------------- #


def test_refilled_history_upgrades_the_final_grounding_and_assurance() -> None:
    """基线只有规范证据（knowledge）→ 回填带回图谱历史案例 → 终局升为 `history_backed`。"""
    sink = _Sink()
    _ctx, result = _run(_ScriptedRetriever(base_source="standard"), _ReviseLLM(), sink)

    (gap,) = result.evidence_requests
    assert gap.satisfied_evidence
    assert result.grounding.basis == "history", "终局的依据等级必须反映回填后的证据"
    assert result.assurance.level == "history_backed"
    assert "grounding_reassessed" in sink.names(), "等级变化必须落事件（P5）"
    assert "依据等级：history_backed" in result.conclusion


def test_an_unrefilled_run_does_not_claim_history_backed() -> None:
    """对照组：没有回填 → 等级维持 knowledge，不得凭空升格。"""
    _ctx, result = _run(_ScriptedRetriever(base_source="standard", fail_gap=True), _ReviseLLM())

    assert result.grounding.basis == "knowledge"
    assert result.assurance.level == "knowledge_based"
    assert "依据等级：knowledge_based" in result.conclusion
