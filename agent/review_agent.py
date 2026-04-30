"""ReviewAgent - PRD 和原型图审查 Agent"""

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from models.minimax import MiniMaxModel
from agent.tools import get_all_tools
from agent.prompts import REVIEW_AGENT_PROMPT


class ReviewAgent:
    """PRD & 原型图审查 Agent"""

    def __init__(self):
        # 初始化 MiniMax 模型
        self.model = MiniMaxModel()

        # 创建 ReAct Agent
        self.agent = create_react_agent(
            model=self.model.chat,
            tools=get_all_tools(),
            state_modifier=REVIEW_AGENT_PROMPT,
            checkpointer=MemorySaver(),
        )

    def invoke(self, input: str, config: dict = None) -> dict:
        """同步调用 Agent"""
        if config is None:
            config = {}

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


def create_review_agent() -> ReviewAgent:
    """工厂函数：创建 ReviewAgent"""
    return ReviewAgent()