"""Conversational CLI.

One ``SessionState`` lives for the whole process; every turn builds a **fresh,
stateless run** (fresh evidence registry, fresh RunContext). That is the
"Run stateless / Session stateful" rule made concrete: nothing leaks between
turns except the explicit session snapshot.

The LLM and the retriever are built **once** per process and shared across
turns. Building them per turn used to open a new Neo4j driver and a new httpx
client every time, and nothing ever closed them.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from .config import settings
from .contracts import HumanDecision, HumanProvidedFact, RunInput, TurnSummary
from .llm import FakeLLM
from .memory import EvidenceRegistry
from .observability import JsonlEventLog
from .ports import RunContext
from .rag import InMemoryRetriever
from .session import SessionState
from .workflow import run


def _build_llm(offline: bool):
    if offline:
        return FakeLLM(
            {
                1: {"E01": "revise", "E02": "approve", "E03": "reject", "E04": "approve",
                    "E05": "revise", "E06": "approve"},
                2: {e: "approve" for e in ("E01", "E02", "E03", "E04", "E05", "E06")},
            }
        )
    from .llm import DeepSeekLLM

    return DeepSeekLLM()


def _build_retriever(offline: bool, mode: str | None):
    """返回 ``(retriever, notes)``；``notes`` 里的降级说明必须被打印出来。"""
    if offline and (mode is None or mode in {"auto", "memory"}):
        return InMemoryRetriever(), ["backend=memory｜--offline：不访问 Qdrant / Neo4j"]

    from .rag_llama.factory import build_retriever

    return build_retriever(settings, mode=mode, offline=offline)


async def _make_human_callback(review):
    """Front-loaded HITL. Blocking ``input`` runs in a worker thread."""
    print("\n" + "=" * 72)
    print("⚠  未检索到历史案例，本次结论仅基于规范与通用工程知识")
    print("=" * 72)
    print(f"系统理解：{review.understood_request}")
    print(f"缺失信息：{'、'.join(review.missing)}")
    print(f"建议专家：{'、'.join(review.candidate_experts)}")
    for line in review.questions:
        print(f"  ? {line}")

    answer = await asyncio.to_thread(input, "接受默认判断并继续？[Y/n] ")
    if answer.strip().lower() in {"n", "no"}:
        return HumanDecision(proceed=False)

    extra = await asyncio.to_thread(input, "补充见解（留空跳过）: ")
    facts = [HumanProvidedFact(raw_text=extra.strip())] if extra.strip() else []
    return HumanDecision(
        proceed=True,
        approved_experts=list(review.candidate_experts),
        provided_facts=facts,
    )


async def _one_turn(
    session: SessionState,
    request: str,
    llm,
    retriever,
    retriever_notes: list[str],
    verbose: bool,
) -> None:
    run_id = uuid.uuid4().hex[:12]
    events = JsonlEventLog(settings.events_path, run_id)
    ctx = RunContext(
        run_id=run_id,
        llm=llm,
        registry=EvidenceRegistry(),
        events=events,
        retriever=retriever,
        ask_human=_make_human_callback,
    )
    # 检索后端的每一次降级都要落在事件日志里，否则「查不到」和
    # 「压根没查」在事后无法区分（design.md P5）。
    events.emit("retriever_selected", resolved=retriever_notes[0], notes=list(retriever_notes))
    if verbose:
        print(f"[run {run_id}] 会话第 {len(session.turns) + 1} 轮")

    try:
        result = await run(RunInput(request=request, session=session.snapshot()), ctx)
    finally:
        events.close()

    print("\n" + result.conclusion)
    if verbose:
        print(f"\n[tokens] {result.usage.model_dump()}")
        print(f"[警告] {list(result.warnings)}")
        print(f"[检索] {retriever_notes[0]}")

    session.append(
        TurnSummary(
            turn=len(session.turns) + 1,
            user_request=request,
            conclusion=result.conclusion.splitlines()[0] if result.conclusion else "",
            confirmed_experts=result.active_experts,
            evidence_ids=result.evidence_ids,
            assurance=result.assurance,
        )
    )


async def _repl(
    offline: bool,
    verbose: bool,
    once: str | None,
    rag_mode: str | None,
) -> None:
    session = SessionState()
    llm = _build_llm(offline)
    retriever, notes = _build_retriever(offline, rag_mode)

    print("检索后端：")
    for note in notes:
        print(f"  · {note}")
    if len(notes) > 1:
        print("  ⚠ 发生了降级：上面的顺序就是回退路径。")
    print()

    try:
        if once:
            await _one_turn(session, once, llm, retriever, notes, verbose)
            return

        print("工程变更方案生成 · 实验版（输入 exit 退出）")
        while True:
            try:
                request = (await asyncio.to_thread(input, "\n变更请求> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not request:
                continue
            if request.lower() in {"exit", "quit"}:
                break
            if session.pending_run_id:
                print("上一个请求仍在等待确认，请先完成它。")
                continue
            await _one_turn(session, request, llm, retriever, notes, verbose)
    finally:
        _close(retriever)
        # LLM 适配器持有异步 httpx 客户端；在同一个事件循环里优雅关闭。
        acloser = getattr(llm, "aclose", None)
        if callable(acloser):
            await acloser()


def _close(target) -> None:
    closer = getattr(target, "close", None)
    if callable(closer):
        closer()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="工程变更方案生成（实验版）")
    parser.add_argument("--offline", action="store_true", help="使用 FakeLLM + 内存检索（不需要网络/密钥）")
    parser.add_argument("--request", "-r", help="单次请求后退出")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--rag",
        choices=["auto", "llamaindex", "graph", "memory"],
        default=None,
        help="检索后端（默认取 RAG_BACKEND，通常为 auto）",
    )
    args = parser.parse_args(argv)

    try:
        asyncio.run(_repl(args.offline, not args.quiet, args.request, args.rag))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
