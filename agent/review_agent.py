"""ReviewAgent - PRD 和原型图审查 Agent"""

from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from models.minimax import MiniMaxModel
from agent.tools import get_all_tools
from agent.prompts import REVIEW_AGENT_PROMPT


class StreamChunkType(str, Enum):
    """流式输出的类型"""
    THINKING = "thinking"      # AI 思考/推理过程
    TOOL_CALL = "tool_call"   # 工具调用
    TOOL_RESULT = "tool_result"  # 工具返回结果
    FINAL = "final"           # 最终回复
    STATUS = "status"         # 状态信息（如进入某个节点）


@dataclass
class StreamChunk:
    """流式输出Chunk"""
    type: StreamChunkType
    content: str | dict  # thinking/final 是 str，tool_call/tool_result 是 dict
    node: str = ""       # 来源节点：model, tools, 等

    def __str__(self) -> str:
        prefix = f"[{self.type.value}]"
        if self.node:
            prefix += f" [{self.node}]"
        if isinstance(self.content, str):
            return f"{prefix} {self.content}"
        return f"{prefix} {self.content}"


class ReviewAgent:
    """PRD & 原型图审查 Agent"""

    def __init__(self):
        # 初始化 MiniMax 模型
        self.model = MiniMaxModel()

        # 创建 ReAct Agent
        self.agent = create_agent(
            model=self.model.chat,
            tools=get_all_tools(),
            system_prompt=REVIEW_AGENT_PROMPT,
            checkpointer=MemorySaver(),
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

    async def ainvoke(self, input: str, config: dict = None) -> dict:
        """异步调用 Agent"""
        if config is None:
            config = {}

        result = await self.agent.ainvoke(
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
        config = {"stream_mode": "messages"}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}

        # 工具名称映射（用于识别工具调用）
        tool_names = {tool.name for tool in get_all_tools()}

        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": input}]},
            config,
        ):
            chunk = self._parse_stream_event(event, tool_names)
            if chunk:
                yield chunk

    async def astream(self, input: str, thread_id: str = None):
        """
        异步流式调用 Agent，区分输出类型

        Yields:
            StreamChunk: 带有类型标识的流式输出
        """
        config = {"stream_mode": "messages"}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}

        tool_names = {tool.name for tool in get_all_tools()}

        async for event in self.agent.astream(
            {"messages": [{"role": "user", "content": input}]},
            config,
        ):
            chunk = self._parse_stream_event(event, tool_names)
            if chunk:
                yield chunk

    def _parse_stream_event(self, event: dict, tool_names: set) -> StreamChunk | None:
        """
        解析流式事件，识别类型

        Args:
            event: LangGraph stream 事件
            tool_names: 可用工具名称集合

        Returns:
            StreamChunk 或 None（跳过无意义的空事件）
        """
        # 处理 messages 模式的流式输出
        if "messages" in event:
            messages_batch = event["messages"]
            if not messages_batch:
                return None

            # 获取最后一条消息
            last_msg = messages_batch[-1]

            # AIMessage - 可能是思考或最终回复
            if hasattr(last_msg, "type") and last_msg.type == "ai":
                content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

                # 检查是否有工具调用（推理过程中的思考）
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_call = last_msg.tool_calls[0]
                    return StreamChunk(
                        type=StreamChunkType.THINKING,
                        content=content,
                        node="model",
                    )

                # 有内容但没有工具调用 - 可能是最终回复或中间思考
                if content:
                    # 空内容通常是中间步骤
                    if not content.strip():
                        return None
                    return StreamChunk(
                        type=StreamChunkType.FINAL,
                        content=content,
                        node="model",
                    )

            # ToolMessage - 工具返回结果
            if hasattr(last_msg, "type") and last_msg.type == "tool":
                tool_name = last_msg.name if hasattr(last_msg, "name") else "unknown"
                content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                return StreamChunk(
                    type=StreamChunkType.TOOL_RESULT,
                    content={"tool": tool_name, "result": content},
                    node="tools",
                )

            return None

        # 处理 updates 模式的输出
        for node_name, node_data in event.items():
            if node_name == "__root__":
                continue

            # model 节点的输出 - AI 推理
            if node_name == "model":
                messages = node_data.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        tool_call = last_msg.tool_calls[0]
                        return StreamChunk(
                            type=StreamChunkType.THINKING,
                            content=f"调用工具: {tool_call.get('name', 'unknown')}",
                            node=node_name,
                        )
                    if hasattr(last_msg, "content") and last_msg.content:
                        return StreamChunk(
                            type=StreamChunkType.THINKING,
                            content=last_msg.content,
                            node=node_name,
                        )

            # tools 节点的输出 - 工具执行结果
            elif node_name == "tools":
                messages = node_data.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "type") and msg.type == "tool":
                        tool_name = getattr(msg, "name", "unknown")
                        content = getattr(msg, "content", "")
                        return StreamChunk(
                            type=StreamChunkType.TOOL_RESULT,
                            content={"tool": tool_name, "result": content[:500]},
                            node=node_name,
                        )

            # 其他节点 - 状态信息
            else:
                return StreamChunk(
                    type=StreamChunkType.STATUS,
                    content=f"进入节点: {node_name}",
                    node=node_name,
                )

        return None


def create_review_agent() -> ReviewAgent:
    """工厂函数：创建 ReviewAgent"""
    return ReviewAgent()