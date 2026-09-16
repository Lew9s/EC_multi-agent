"""LlamaIndex LLM 桥接。

LlamaIndex 需要的是它自己的 ``LLM`` 接口，而本项目的 LLM 在 ``ports.LLMPort``
后面（``DeepSeekLLM`` 自带缓存、重试、用量统计；``FakeLLM`` 离线确定性）。

这里做 **适配而不是替换**：``LlamaLLMBridge`` 把 ``LLMPort`` 包成
``CustomLLM``。好处是 LlamaIndex 内部的每一次调用（例如知识抽取）
依然会走项目自己的缓存与用量统计，key 也依然只存在于 ``config`` 与
``DeepSeekLLM`` 内部 —— LlamaIndex 拿不到它。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from pydantic import PrivateAttr

from ..config import Settings
from ..config import settings as default_settings
from ..ports import LLMPort

# LlamaIndex 的 completion 接口只给一个 prompt 字符串，没有单独的 system 位。
_SYSTEM = (
    "你是造船工程变更领域的知识抽取助手。"
    "只输出符合要求的 JSON，不要输出任何解释、注释或 Markdown 代码块。"
)


def run_coroutine_blocking(coro: Any) -> Any:
    """在同步上下文里跑一个协程。

    LlamaIndex 的部分抽取器是同步接口（内部用 ``asyncio.run``），
    而我们的端口是异步的。两条路径都要能走：

    * 当前线程没有事件循环（工作线程里跑摄取）-> 直接 ``asyncio.run``；
    * 当前线程**已有**运行中的循环 -> 换一个线程 + 新循环，绝不 ``run`` 嵌套。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class LlamaLLMBridge(CustomLLM):
    """``LLMPort`` -> LlamaIndex ``CustomLLM``。"""

    model_name: str = "deepseek-flash"
    context_window: int = 65536
    num_output: int = 2048

    _port: LLMPort = PrivateAttr()
    _purpose: str = PrivateAttr(default="llama")

    def __init__(self, port: LLMPort, *, purpose: str = "llama", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._port = port
        self._purpose = purpose

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            context_window=self.context_window,
            num_output=self.num_output,
            is_chat_model=False,
            is_function_calling_model=False,
        )

    def _finish(self, result: Any) -> CompletionResponse:
        return CompletionResponse(
            text=result.content,
            # raw 只放非敏感的用量信息，不放 prompt 原文 / 密钥。
            raw={"model": result.model, "cached": result.cached},
        )

    # -- sync --------------------------------------------------------------- #
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        del formatted, kwargs
        return self._finish(
            run_coroutine_blocking(
                self._port.complete(purpose=self._purpose, system=_SYSTEM, user=prompt)
            )
        )

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        # 本项目不做流式：抽取与专家意见都是「一次性结构化输出」。
        raise NotImplementedError("LlamaLLMBridge 不支持流式生成（本项目不需要）")

    # -- async -------------------------------------------------------------- #
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        del formatted, kwargs
        return self._finish(
            await self._port.complete(purpose=self._purpose, system=_SYSTEM, user=prompt)
        )

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> AsyncGenerator[CompletionResponse, None]:
        raise NotImplementedError("LlamaLLMBridge 不支持流式生成（本项目不需要）")
        yield  # pragma: no cover  — 让函数体成为异步生成器


def build_llama_llm(
    cfg: Settings | None = None,
    *,
    offline: bool = False,
    purpose: str = "llama",
) -> LlamaLLMBridge:
    """给 LlamaIndex 用的 LLM。

    ``offline=True`` 时包 ``FakeLLM``，于是整条 LlamaIndex 摄取链路
    （包括知识抽取）都能在没有 key 的情况下被测试。
    """
    cfg = cfg or default_settings
    if offline:
        from ..llm import FakeLLM

        port: LLMPort = FakeLLM()
    else:
        from ..llm import DeepSeekLLM

        port = DeepSeekLLM()
    return LlamaLLMBridge(
        port,
        purpose=purpose,
        model_name="fake" if offline else cfg.deepseek_model,
        context_window=8192 if offline else 65536,
        num_output=cfg.llm_max_tokens,
    )
