"""Anthropic ReviewAgent - 使用 MiniMax Anthropic 原生流式接口"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, Iterator, Callable

from models.minimax_anthropic import (
    MiniMaxAnthropicModel,
    AnthropicStreamChunk,
    StreamChunkType as AnthropicChunkType,
)
from agent.tools import get_all_tools
from agent.prompts import REVIEW_AGENT_PROMPT


class StreamChunkType(str, Enum):
    """流式输出的语义类型"""
    THINKING = "thinking"       # AI 推理思考过程
    TEXT = "text"              # 文本回复
    TOOL_CALL = "tool_call"    # 工具调用
    TOOL_RESULT = "tool_result" # 工具返回结果
    FINAL = "final"            # 最终回复
    STATUS = "status"          # 状态信息


@dataclass
class StreamChunk:
    """流式输出Chunk"""
    type: StreamChunkType
    content: str | dict
    node: str = "anthropic"

    def __str__(self) -> str:
        prefix = f"[{self.type.value}]"
        if self.node:
            prefix += f" [{self.node}]"
        if isinstance(self.content, str):
            return f"{prefix} {self.content}"
        return f"{prefix} {self.content}"


class AnthropicReviewAgent:
    """
    PRD & 原型图审查 Agent - Anthropic 原生流式版本

    实现自己的 ReAct 循环，支持：
    - 原生 thinking 流式输出
    - text 流式输出
    - 工具调用（tool_use）
    """

    def __init__(self, model_name: str = "MiniMax-M2.7"):
        # 初始化 MiniMax Anthropic 模型
        self.model = MiniMaxAnthropicModel(
            model=model_name,
            thinking_enabled=True,
        )

        # 工具列表
        self.tools = get_all_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}

        # 消息历史
        self.messages = []

    def _format_tools(self) -> list[dict]:
        """将 LangChain tools 格式化为 Anthropic tools 格式"""
        formatted = []
        for tool in self.tools:
            name = tool.name
            desc = getattr(tool, 'description', '') or name

            # 尝试获取参数模式
            args_schema = getattr(tool, 'args_schema', None)
            if args_schema:
                # 尝试获取 JSON schema
                if hasattr(args_schema, 'model_json_schema'):
                    schema = args_schema.model_json_schema()
                elif hasattr(args_schema, 'schema'):
                    schema = args_schema.schema
                else:
                    schema = {"type": "object", "properties": {}}
            else:
                schema = {"type": "object", "properties": {}}

            formatted.append({
                "name": name,
                "description": desc,
                "input_schema": schema
            })
        return formatted

    def _parse_tool_calls(self, response) -> list[dict]:
        """从响应中解析工具调用"""
        tool_calls = []
        for block in response.content:
            if block.type == "tool_use":
                tool_name = getattr(block, 'name', '')
                tool_input = getattr(block, 'input', {}) or {}
                tool_id = getattr(block, 'id', '')
                tool_calls.append({
                    "name": tool_name,
                    "input": tool_input,
                    "id": tool_id
                })
        return tool_calls

    def invoke(self, input: str, thread_id: str = None) -> dict:
        """
        同步调用 Agent（返回完整消息）
        """
        self.messages = [{"role": "user", "content": input}]

        # 格式化 tools
        tools_formatted = self._format_tools()
        self.model.tools = tools_formatted

        response = self.model.invoke(self.messages)

        # 解析工具调用
        tool_calls = self._parse_tool_calls(response)

        # 如果有工具调用，执行它们
        while tool_calls:
            # 添加助手的工具调用消息
            for tc in tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "name": tc["name"],
                        "input": tc["input"],
                        "id": tc["id"]
                    }]
                })

                # 执行工具
                tool_result = self._execute_tool(tc)
                # 添加工具结果消息
                self.messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": tool_result
                    }]
                })

            # 继续调用模型
            response = self.model.invoke(self.messages)
            tool_calls = self._parse_tool_calls(response)

        # 返回最终结果
        result_content = []
        for block in response.content:
            if hasattr(block, 'thinking') and block.thinking:
                result_content.append(f"[思考] {block.thinking}")
            if hasattr(block, 'text') and block.text:
                result_content.append(block.text)

        return {"messages": [{"content": "\n\n".join(result_content)}]}

    def _execute_tool(self, tool_call: dict) -> str:
        """执行工具调用"""
        tool_name = tool_call.get("name", "")
        tool_input = tool_call.get("input", {})

        tool = self.tool_map.get(tool_name)
        if not tool:
            return f"Error: Unknown tool {tool_name}"

        try:
            # 执行工具
            result = tool.invoke(tool_input)
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def stream(self, input: str, thread_id: str = None) -> Iterator[StreamChunk]:
        """
        同步流式调用 Agent，区分输出类型

        实现简单的 ReAct 循环：
        1. 调用模型获取响应
        2. 如果有工具调用，执行并流式返回结果
        3. 继续直到没有工具调用
        """
        self.messages = [{"role": "user", "content": input}]
        tools_formatted = self._format_tools()
        self.model.tools = tools_formatted

        # 用于累积完整内容
        thinking_buffer = []
        text_buffer = []
        final_text = []

        # 第一次调用 - 流式输出 thinking 和 text
        for chunk in self.model.stream(self.messages):
            if chunk.type == AnthropicChunkType.THINKING:
                thinking_buffer.append(chunk.content)
                yield StreamChunk(
                    type=StreamChunkType.THINKING,
                    content=chunk.content,
                    node="model",
                )
            elif chunk.type == AnthropicChunkType.TEXT:
                text_buffer.append(chunk.content)
                final_text.append(chunk.content)
                yield StreamChunk(
                    type=StreamChunkType.TEXT,
                    content=chunk.content,
                    node="model",
                )
            elif chunk.type == AnthropicChunkType.TOOL_CALL:
                yield StreamChunk(
                    type=StreamChunkType.TOOL_CALL,
                    content=chunk.content,
                    node="model",
                )
            elif chunk.type == AnthropicChunkType.MESSAGE_END:
                # 消息结束，检查是否有工具调用
                pass

        # 获取完整响应
        # 注意：流式结束后需要重新调用以获取完整响应进行工具调用
        response = self.model.invoke(self.messages)
        tool_calls = self._parse_tool_calls(response)

        # 处理工具调用
        while tool_calls:
            for tc in tool_calls:
                # 格式化工具参数用于显示
                tool_input_str = json.dumps(tc['input'], ensure_ascii=False)
                yield StreamChunk(
                    type=StreamChunkType.TOOL_CALL,
                    content=f"调用工具: {tc['name']}\n参数: {tool_input_str}",
                    node="model",
                )

                # 执行工具
                tool_result = self._execute_tool(tc)

                yield StreamChunk(
                    type=StreamChunkType.TOOL_RESULT,
                    content={"tool": tc["name"], "result": tool_result},
                    node="tools",
                )

                # 添加消息到历史
                self.messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "name": tc["name"],
                        "input": tc["input"],
                        "id": tc["id"]
                    }]
                })
                self.messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": tool_result
                    }]
                })

            # 继续调用模型
            for chunk in self.model.stream(self.messages):
                if chunk.type == AnthropicChunkType.THINKING:
                    thinking_buffer.append(chunk.content)
                    yield StreamChunk(
                        type=StreamChunkType.THINKING,
                        content=chunk.content,
                        node="model",
                    )
                elif chunk.type == AnthropicChunkType.TEXT:
                    text_buffer.append(chunk.content)
                    final_text.append(chunk.content)
                    yield StreamChunk(
                        type=StreamChunkType.TEXT,
                        content=chunk.content,
                        node="model",
                    )
                elif chunk.type == AnthropicChunkType.MESSAGE_END:
                    pass

            response = self.model.invoke(self.messages)
            tool_calls = self._parse_tool_calls(response)

        # 最终输出
        if final_text:
            yield StreamChunk(
                type=StreamChunkType.FINAL,
                content="".join(final_text),
                node="model",
            )

    async def ainvoke(self, input: str, config: dict = None) -> dict:
        """异步调用 Agent"""
        return self.invoke(input)

    async def astream(self, input: str, thread_id: str = None) -> AsyncGenerator[StreamChunk, None]:
        """异步流式调用 Agent"""
        for chunk in self.stream(input, thread_id):
            yield chunk


def create_anthropic_review_agent(model_name: str = "MiniMax-M2.7") -> AnthropicReviewAgent:
    """工厂函数：创建 Anthropic ReviewAgent"""
    return AnthropicReviewAgent(model_name=model_name)
