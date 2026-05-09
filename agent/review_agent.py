"""ReviewAgent - PRD 和原型图审查 Agent"""

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Union

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from models.minimax import MiniMaxModel
from agent.tools import get_all_tools
from agent.prompts import REVIEW_AGENT_PROMPT


def _split_thinking(content: str) -> list[tuple[str, bool]]:
    """拆分思考标签内容

    Args:
        content: 原始消息内容

    Returns:
        [(text, is_thinking), ...] - is_thinking=True 表示是<think>...</think>之间的内容
    """
    import re
    parts = []
    # 匹配<think>和</think>之间的内容
    pattern = r'<think>(.*?)</think>'
    last_end = 0

    for match in re.finditer(pattern, content, re.DOTALL):
        # 处理标签之前的普通文本
        if match.start() > last_end:
            text_before = content[last_end:match.start()].strip()
            if text_before:
                parts.append((text_before, False))
        # 处理思考标签内容
        thinking_text = match.group(1).strip()
        if thinking_text:
            parts.append((thinking_text, True))
        last_end = match.end()

    # 处理最后一个</think>之后的文本
    if last_end < len(content):
        text_after = content[last_end:].strip()
        if text_after:
            parts.append((text_after, False))

    return parts if parts else [(content, False)]


class StreamChunkType(str, Enum):
    """流式输出的类型"""
    THINKING = "thinking"      # AI 思考/推理过程
    TOOL_CALL = "tool_call"   # 工具调用
    TOOL_RESULT = "tool_result"  # 工具返回结果
    FINAL = "final"           # 最终回复
    STATUS = "status"         # 状态信息（如进入某个节点）
    ERROR = "error"           # 错误信息（如未授权的工具调用）


@dataclass
class StreamChunk:
    """流式输出Chunk"""
    node: str
    messages: list[AIMessage | ToolMessage] = field(default_factory=list)
    is_split: bool = False  # 是否被拆分过

    @property
    def type(self) -> StreamChunkType:
        """从 messages 推导 chunk 类型"""
        if not self.messages:
            return StreamChunkType.STATUS
        first = self.messages[0]
        if isinstance(first, ToolMessage):
            return StreamChunkType.TOOL_RESULT
        if hasattr(first, "tool_calls") and first.tool_calls:
            return StreamChunkType.TOOL_CALL
        return StreamChunkType.THINKING

    @property
    def content(self) -> str:
        """方便获取文本内容"""
        if not self.messages:
            return ""
        return self.messages[0].content or ""

    def __str__(self) -> str:
        prefix = f"[{self.type.value}]"
        if self.node:
            prefix += f" [{self.node}]"
        return f"{prefix} {self.content[:100]}{'...' if len(self.content) > 100 else ''}"


@dataclass
class SplitStreamChunk(StreamChunk):
    """拆分后的 Chunk，继承自 StreamChunk"""
    id: str = ""       # 对应原始 message.id
    chunk_type: str = ""  # "thinking" | "responding"

    def __str__(self) -> str:
        prefix = f"[{self.chunk_type}]"
        if self.node:
            prefix += f" [{self.node}]"
        return f"{prefix} {self.content[:100]}{'...' if len(self.content) > 100 else ''}"


class ReviewAgent:
    """PRD & 原型图审查 Agent"""

    def __init__(self):
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_data")
        os.makedirs(checkpoint_dir, exist_ok=True)
        db_path = os.path.join(checkpoint_dir, "checkpoints.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        self.sync_checkpointer = SqliteSaver(conn)
        self.db_path = db_path

        self.model = MiniMaxModel()

        self.agent = create_agent(
            model=self.model.chat,
            tools=get_all_tools(),
            system_prompt=REVIEW_AGENT_PROMPT,
            checkpointer=self.sync_checkpointer,
        )

    def invoke(self, input: str, thread_id: str = None) -> dict:
        """同步调用 Agent"""
        config = {}
        if thread_id:
            config = {"configurable": {"thread_id": thread_id}}

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": input}]},
            config,
        )
        return result

    async def ainvoke(self, input: str, config: dict = None, thread_id: str = None) -> dict:
        """异步调用 Agent"""
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        if config is None:
            config = {"configurable": {"thread_id": thread_id}}
        elif "configurable" not in config:
            config["configurable"] = {"thread_id": thread_id}

        async with AsyncSqliteSaver.from_conn_string(self.db_path) as async_checkpointer:
            async_agent = create_agent(
                model=self.model.chat,
                tools=get_all_tools(),
                system_prompt=REVIEW_AGENT_PROMPT,
                checkpointer=async_checkpointer,
            )
            result = await async_agent.ainvoke(
                {"messages": [{"role": "user", "content": input}]},
                config,
            )
        return result

    def stream(self, input: str, thread_id: str = None):
        """
        同步流式调用 Agent，区分输出类型

        Yields:
            StreamChunk: 带有类型标识的流式输出
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        config = {"stream_mode": "updates", "configurable": {"thread_id": thread_id}}

        tool_names = {tool.name for tool in get_all_tools()}

        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": input}]},
            config,
        ):
            print(f"[DEBUG] event type: {type(event)}, event: {repr(event)[:200]}")
            chunk = self._parse_stream_event(event, tool_names)
            if chunk:
                if isinstance(chunk, list):
                    yield from chunk
                else:
                    yield chunk

    async def astream(self, input: str, thread_id: str = None):
        """
        异步流式调用 Agent，区分输出类型

        Yields:
            StreamChunk: 带有类型标识的流式输出
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        config = {"stream_mode": "updates", "configurable": {"thread_id": thread_id}}

        tool_names = {tool.name for tool in get_all_tools()}

        async with AsyncSqliteSaver.from_conn_string(self.db_path) as async_checkpointer:
            async_agent = create_agent(
                model=self.model.chat,
                tools=get_all_tools(),
                system_prompt=REVIEW_AGENT_PROMPT,
                checkpointer=async_checkpointer,
            )
            async for event in async_agent.astream(
                {"messages": [{"role": "user", "content": input}]},
                config,
            ):
                chunk = self._parse_stream_event(event, tool_names)
                if chunk:
                    if isinstance(chunk, list):
                        for c in chunk:
                            yield c
                    else:
                        yield chunk

    def _parse_stream_event(self, event: dict | tuple, tool_names: set) -> StreamChunk | list[StreamChunk] | None:
        """
        解析流式事件，识别类型

        Args:
            event: LangGraph stream 事件（updates模式为dict，messages模式为tuple）
            tool_names: 可用工具名称集合

        Returns:
            StreamChunk、StreamChunk列表 或 None（当content包含<think>标签时返回列表）
        """
        # 处理 messages 模式的流式输出
        #####################################################################################################
        # if "messages" in event:
        #     messages_batch = event["messages"]
        #     if not messages_batch:
        #         return None

        #     # 获取最后一条消息
        #     last_msg = messages_batch[-1]

        #     # AIMessage - 可能是思考或最终回复
        #     if hasattr(last_msg, "type") and last_msg.type == "ai":
        #         content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        #         # 检查是否有工具调用
        #         if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        #             tool_call = last_msg.tool_calls[0]
        #             tool_name = tool_call.get('name', 'unknown')

        #             if tool_name not in tool_names:
        #                 return StreamChunk(
        #                     type=StreamChunkType.ERROR,
        #                     content=f"未授权的工具调用: {tool_name}",
        #                     node="model",
        #                 )

        #             tool_args = tool_call.get('args', {})
        #             args_str = ", ".join(
        #                 f"{k}={repr(v)[:50]}{'...' if len(str(v)) > 50 else ''}"
        #                 for k, v in tool_args.items()
        #             )
        #             return StreamChunk(
        #                 type=StreamChunkType.TOOL_CALL,
        #                 content=f"{tool_name}({args_str})" if args_str else tool_name,
        #                 node="model",
        #             )

        #         # 有内容但没有工具调用 - 可能是最终回复或中间思考
        #         if content:
        #             # 空内容通常是中间步骤
        #             if not content.strip():
        #                 return None
        #             return StreamChunk(
        #                 type=StreamChunkType.FINAL,
        #                 content=content,
        #                 node="model",
        #             )

        #     # ToolMessage - 工具返回结果
        #     if hasattr(last_msg, "type") and last_msg.type == "tool":
        #         tool_name = last_msg.name if hasattr(last_msg, "name") else "unknown"
        #         content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        #         return StreamChunk(
        #             type=StreamChunkType.TOOL_RESULT,
        #             content={"tool": tool_name, "result": content},
        #             node="tools",
        #         )

        #     return None

        # 处理 updates 模式的输出
        print(f"{'='*80}\n")
        
        for node_name, node_data in event.items():
            if node_name == "__root__":
                continue

            # model 节点的输出 - AI 推理
            if node_name == "model":
                messages = node_data.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            # 验证所有工具调用
                            for tool_call in last_msg.tool_calls:
                                tool_name = tool_call.get('name', 'unknown')
                                if tool_name not in tool_names:
                                    return StreamChunk(
                                        node=node_name,
                                        messages=[last_msg],
                                    )
                                for k,v in tool_call.get('args', {}).items():
                                    print(f"[工具调用具名：{tool_name}\n 参数内容：{k}={repr(v)}")
                            return StreamChunk(
                                node=node_name,
                                messages=[last_msg],
                            )
                        else:
                            # 检查是否包含思考标签
                            parts = _split_thinking(last_msg.content)
                            if len(parts) == 1 and not parts[0][1]:
                                # 没有思考标签，直接返回
                                return StreamChunk(
                                    node=node_name,
                                    messages=[last_msg],
                                )
                            else:
                                # 拆分思考内容，返回多个 SplitStreamChunk
                                chunks = []
                                for part_content, is_thinking in parts:
                                    # 创建覆写了 content 的新消息
                                    new_msg = last_msg.model_copy() if hasattr(last_msg, 'model_copy') else last_msg
                                    new_msg.content = part_content
                                    chunks.append(SplitStreamChunk(
                                        node=node_name,
                                        messages=[new_msg],
                                        is_split=True,
                                        id=last_msg.id,
                                        chunk_type="thinking" if is_thinking else "responding",
                                    ))
                                return chunks
                        
            # tools 节点的输出 - 工具执行结果
            elif node_name == "tools":
                tool_messages = node_data.get("messages", [])
                return StreamChunk(
                    node=node_name,
                    messages=[msg for msg in tool_messages if hasattr(msg, "type") and msg.type == "tool"],
                )

            # 其他节点 - 状态信息
            else:
                return StreamChunk(
                    node=node_name,
                    messages=[],
                )

        return None


def create_review_agent() -> ReviewAgent:
    """工厂函数：创建 ReviewAgent"""
    return ReviewAgent()