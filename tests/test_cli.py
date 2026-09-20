"""CLI 的交互行为：退出命令的识别，以及给人看的专家标识渲染。

两条性质在这里被钉住：退出命令不产生任何 run；专家标识只在**展示**时换成
专业名，回传给 workflow 的仍是标识本身（``HumanDecision.approved_experts``）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from ec_renew.contracts import HumanReviewRequest
from ec_renew.interface import cli


def _feed(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> None:
    """按顺序供给 ``input()`` 的返回值。"""
    remaining: Iterator[str] = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: next(remaining))


@pytest.mark.parametrize("command", ["/exit", "/quit", "exit", "quit"])
def test_exit_command_ends_repl_without_running_a_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    _feed(monkeypatch, [command])

    asyncio.run(cli._repl(offline=True, verbose=True, once=None, rag_mode=None))

    out = capsys.readouterr().out
    assert "输入 /exit 退出" in out
    assert "[run " not in out


def test_suggested_experts_are_shown_by_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _feed(monkeypatch, ["y", ""])
    review = HumanReviewRequest(
        understood_request="301分段FR36污水井更换加厚板",
        candidate_experts=["E01", "E03", "E06"],
        questions=["是否确认按上述专家范围评估？"],
    )

    decision = asyncio.run(cli._make_human_callback(review))

    out = capsys.readouterr().out
    assert "建议专家：结构设计、质量规范、材料与焊接" in out
    assert "E01" not in out
    # 展示用专业名，契约仍走标识：人类批准的专家集不受渲染影响。
    assert decision.proceed is True
    assert decision.approved_experts == ["E01", "E03", "E06"]


def test_unknown_expert_id_is_shown_verbatim() -> None:
    """职业目录里没有的标识不静默丢失，原样透出以便被发现。"""
    assert cli._expert_labels(["E01", "E99"]) == "结构设计、E99"
