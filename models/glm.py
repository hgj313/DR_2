"""GLM 模型封装 - 用于视觉分析"""

import os
import base64
from typing import Union

GLM_API_KEY = os.getenv("GLM_API_KEY", "2f84b8c9874e45c295e110c43952afce.MmcJgxFdZE5Haqml")
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

class GLMModel:
    """GLM-4.6V-Flash 模型封装，用于视觉理解"""

    def __init__(
        self,
        model: str = "glm-4.6v-flash",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.api_key = GLM_API_KEY
        self.base_url = GLM_BASE_URL
        self.temperature = temperature
        self.max_tokens = max_tokens

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
    ) -> str:
        """分析图片（同步调用）"""
        import httpx

        # 读取图片并转为 base64
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": prompt},
                    ]
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    async def analyze_image_async(
        self,
        image_path: str,
        prompt: str,
    ) -> str:
        """分析图片（异步调用）"""
        import httpx
        import asyncio

        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": prompt},
                    ]
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]