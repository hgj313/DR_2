"""MiniMax 模型封装 - 用于 Agent 主体"""

import os
from langchain_openai import ChatOpenAI

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "sk-cp-HbqEw_in0Ak-rs-fsLEvknbmp6mHtZUPx8O2KEG5tBnwUrIMSRIQzAt88vpowSXBholobeKgTuK_dyRGQMaou5hi02wgaKx0yaw3fs1qcHvxAgHEvCp1Hlk")
MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"

class MiniMaxModel:
    """MiniMax 模型封装，提供 ChatOpenAI 兼容接口"""

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            api_key=MINIMAX_API_KEY,
            base_url=MINIMAX_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def invoke(self, messages: list[dict]) -> str:
        """同步调用"""
        return self.llm.invoke(messages)

    async def ainvoke(self, messages: list[dict]) -> str:
        """异步调用"""
        return await self.llm.ainvoke(messages)

    @property
    def chat(self):
        return self.llm