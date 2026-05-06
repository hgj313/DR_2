"""MiniMax Anthropic 兼容接口 - 支持原生 thinking 流式输出"""

import os
from typing import AsyncGenerator, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum

import anthropic
from anthropic.types import Message

MINIMAX_ANTHROPIC_API_KEY = os.getenv("MINIMAX_API_KEY", "sk-cp-HbqEw_in0Ak-rs-fsLEvknbmp6mHtZUPx8O2KEG5tBnwUrIMSRIQzAt88vpowSXBholobeKgTuK_dyRGQMaou5hi02wgaKx0yaw3fs1qcHvxAgHEvCp1Hlk")
MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimaxi.com/anthropic"


class StreamChunkType(str, Enum):
    """流式输出的语义类型"""
    THINKING = "thinking"       # AI 推理思考过程
    TEXT = "text"             # 文本回复
    TOOL_CALL = "tool_call"   # 工具调用
    TOOL_RESULT = "tool_result"  # 工具返回结果
    MESSAGE_START = "message_start"  # 消息开始
    MESSAGE_DELTA = "message_delta"  # 消息增量
    CONTENT_BLOCK_START = "content_block_start"  # 内容块开始
    CONTENT_BLOCK_DELTA = "content_block_delta"  # 内容块增量
    MESSAGE_END = "message_end"  # 消息结束


@dataclass
class AnthropicStreamChunk:
    """Anthropic 流式输出 Chunk"""
    type: StreamChunkType
    content: str
    node: str = "anthropic"

    def __str__(self) -> str:
        return f"[{self.type.value}] {self.content}"


@dataclass
class AIMessageChunk:
    """LangChain 兼容的 AIMessageChunk"""
    content: str = ""
    tool_call_chunks: list = field(default_factory=list)
    usage_metadata: dict = field(default_factory=dict)

    def __add__(self, other: "AIMessageChunk") -> "AIMessageChunk":
        """支持增量合并"""
        return AIMessageChunk(
            content=self.content + other.content,
            tool_call_chunks=self.tool_call_chunks + other.tool_call_chunks,
            usage_metadata=self.usage_metadata,
        )


class MiniMaxAnthropicModel:
    """
    MiniMax Anthropic 兼容接口模型

    支持：
    - 原生 thinking 流式输出
    - text 流式输出
    - 工具调用（tool_use）
    - LangChain ChatModel 兼容接口
    """

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        temperature: float = 1.0,
        max_tokens: int = 4096,
        thinking_enabled: bool = True,
        tools: list = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_enabled = thinking_enabled
        self.tools = tools or []

        self.client = anthropic.Anthropic(
            api_key=MINIMAX_ANTHROPIC_API_KEY,
            base_url=MINIMAX_ANTHROPIC_BASE_URL,
        )

    def invoke(self, messages: list[dict]) -> Message:
        """同步调用（返回完整消息）"""
        system = ""
        formatted_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                formatted_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # 支持多模态内容
                formatted_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            formatted_content.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "tool_use":
                            formatted_content.append(item)
                    elif isinstance(item, str):
                        formatted_content.append({"type": "text", "text": item})
                formatted_messages.append({"role": role, "content": formatted_content})

        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=formatted_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            thinking={"type": "enabled"} if self.thinking_enabled else None,
            tools=self.tools if self.tools else None,
        )
        return response

    def stream(self, messages: list[dict]) -> Iterator[AnthropicStreamChunk]:
        """
        同步流式调用，返回语义化的流式 Chunk

        Yields:
            AnthropicStreamChunk: 带有类型标识的流式输出
        """
        system = ""
        formatted_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                formatted_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                formatted_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            formatted_content.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "tool_use":
                            formatted_content.append(item)
                    elif isinstance(item, str):
                        formatted_content.append({"type": "text", "text": item})
                formatted_messages.append({"role": role, "content": formatted_content})

        # 调试：打印消息结构
        import logging
        _dbg_logger = logging.getLogger(__name__)
        _dbg_logger.debug(f"[MiniMax] Calling API with {len(formatted_messages)} messages")
        for i, msg in enumerate(formatted_messages):
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, list):
                types = [c.get("type") if isinstance(c, dict) else str(c) for c in content]
                _dbg_logger.debug(f"  [{i}] {role}: {types}")
            else:
                _dbg_logger.debug(f"  [{i}] {role}: {str(content)[:100]}")

        with self.client.messages.stream(
            model=self.model,
            system=system,
            messages=formatted_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            thinking={"type": "enabled"} if self.thinking_enabled else None,
            tools=self.tools if self.tools else None,
        ) as stream:
            for chunk in stream:
                yield from self._parse_stream_chunk(chunk)

    def _parse_stream_chunk(self, chunk) -> Iterator[AnthropicStreamChunk]:
        """解析单个流式 chunk"""
        chunk_type = getattr(chunk, "type", None)

        if chunk_type == "message_start":
            yield AnthropicStreamChunk(
                type=StreamChunkType.MESSAGE_START,
                content="",
            )

        elif chunk_type == "content_block_start":
            content_block = getattr(chunk, "content_block", None)
            if content_block:
                block_type = getattr(content_block, "type", None)
                if block_type == "thinking":
                    yield AnthropicStreamChunk(
                        type=StreamChunkType.THINKING,
                        content="[thinking started]",
                    )
                elif block_type == "text":
                    yield AnthropicStreamChunk(
                        type=StreamChunkType.TEXT,
                        content="[text started]",
                    )
                elif block_type == "tool_use":
                    yield AnthropicStreamChunk(
                        type=StreamChunkType.TOOL_CALL,
                        content="[tool_use started]",
                    )

        elif chunk_type == "content_block_delta":
            delta = getattr(chunk, "delta", None)
            if delta:
                delta_type = getattr(delta, "type", None)

                if delta_type == "thinking_delta":
                    thinking_content = getattr(delta, "thinking", "")
                    if thinking_content:
                        yield AnthropicStreamChunk(
                            type=StreamChunkType.THINKING,
                            content=thinking_content,
                        )

                elif delta_type == "text_delta":
                    text_content = getattr(delta, "text", "")
                    if text_content:
                        yield AnthropicStreamChunk(
                            type=StreamChunkType.TEXT,
                            content=text_content,
                        )

                elif delta_type == "tool_use_delta":
                    # 工具调用增量
                    tool_name = getattr(delta, "name", "")
                    tool_input = getattr(delta, "input", "")
                    if tool_name:
                        yield AnthropicStreamChunk(
                            type=StreamChunkType.TOOL_CALL,
                            content=f"[tool_call: {tool_name}]",
                        )

        elif chunk_type == "message_delta":
            delta = getattr(chunk, "delta", None)
            if delta:
                if hasattr(delta, "stop_reason"):
                    yield AnthropicStreamChunk(
                        type=StreamChunkType.MESSAGE_END,
                        content=f"[stop_reason: {delta.stop_reason}]",
                    )

        elif chunk_type == "message_stop":
            yield AnthropicStreamChunk(
                type=StreamChunkType.MESSAGE_END,
                content="[stream completed]",
            )

    async def ainvoke(self, messages: list[dict]) -> Message:
        """异步调用"""
        return self.invoke(messages)

    async def astream(self, messages: list[dict]) -> AsyncGenerator[AnthropicStreamChunk, None]:
        """
        异步流式调用
        """
        system = ""
        formatted_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                formatted_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                formatted_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            formatted_content.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "tool_use":
                            formatted_content.append(item)
                    elif isinstance(item, str):
                        formatted_content.append({"type": "text", "text": item})
                formatted_messages.append({"role": role, "content": formatted_content})

        async with self.client.messages.stream(
            model=self.model,
            system=system,
            messages=formatted_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            thinking={"type": "enabled"} if self.thinking_enabled else None,
            tools=self.tools if self.tools else None,
        ) as stream:
            async for chunk in stream:
                # Note: async 流式需要 httpx 的 async client
                # Anthropic SDK 自动处理
                yield AnthropicStreamChunk(
                    type=StreamChunkType.MESSAGE_END,
                    content=str(chunk),
                )

    @property
    def chat(self):
        """返回用于 LangChain agent 的 chat 接口"""
        return self


class MiniMaxAnthropicChatModel(MiniMaxAnthropicModel):
    """
    LangChain ChatModel 兼容接口

    提供与 LangChain ChatOpenAI 相似的接口，
    但使用 MiniMax Anthropic 兼容 API
    """

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        temperature: float = 1.0,
        max_tokens: int = 4096,
        thinking_enabled: bool = True,
    ):
        super().__init__(model, temperature, max_tokens, thinking_enabled)

    def _to_langchain_chunk(self, chunk: AnthropicStreamChunk) -> AIMessageChunk:
        """将 AnthropicStreamChunk 转换为 LangChain AIMessageChunk"""
        if chunk.type == StreamChunkType.THINKING:
            return AIMessageChunk(content=chunk.content)
        elif chunk.type == StreamChunkType.TEXT:
            return AIMessageChunk(content=chunk.content)
        elif chunk.type == StreamChunkType.TOOL_CALL:
            return AIMessageChunk(
                content="",
                tool_call_chunks=[{"name": chunk.content, "args": ""}],
            )
        else:
            return AIMessageChunk(content=chunk.content)

    def stream(self, messages: list[dict]) -> Iterator[AIMessageChunk]:
        """
        同步流式调用，返回 LangChain 格式的 AIMessageChunk

        适配 LangGraph 的 stream_mode="messages"
        """
        for anthropic_chunk in super().stream(messages):
            yield self._to_langchain_chunk(anthropic_chunk)
