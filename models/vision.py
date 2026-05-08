"""
视觉模型封装 - 支持多模型对比（Kimi/Qwen）

基于 OpenAI 兼容 API，统一封装视觉理解能力。
测试结果见 prd-test/vision_test/ALIYUN_VISION_MODEL_REPORT.md

支持多 Provider 切换：
- markai: 小平台 (https://api.markai.markqq.com/v1)
- aliyun: 阿里大平台 (https://dashscope.aliyuncs.com/compatible-mode/v1)

阿里云平台测试结果排名（2026-05-07）：
1. kimi-k2.6: 73分 (B级) - 综合最强
2. kimi-k2.5: 72分 (B级) - 格式最规范
3. qwen3.6-max-preview: 71分 (B级) - 细节最丰富（耗时118s）
4. qwen3.6-plus: 64分 (B级) - 平衡之选
5. qwen3.6-flash: 61分 (B级) - 速度最快（23s）
6. qwen3.5-plus: 58分 (C级) - 稳定可用
不支持视觉：MiniMax-M2.5, DeepSeek-V4-Pro, GLM-5(识别异常)
"""

import os
import base64
from typing import Union, Optional, Literal
from dataclasses import dataclass

import httpx


# Provider 配置列表
VISION_PROVIDERS = {
    "markai": {
        "name": "MarkAI (小平台)",
        "api_key": os.getenv("VISION_API_KEY", "sk-_XXxLm0aroXBl-1b5kZitLdSthKIWh8SVEAWJ2aQDKu_edHfsz4meIL1ZFYnJfTF"),
        "base_url": os.getenv("VISION_BASE_URL", "https://api.markai.markqq.com/v1"),
    },
    "aliyun": {
        "name": "阿里大平台",
        "api_key": os.getenv("VISION_ALIYUN_API_KEY", "sk-3e00ee40cade4393b982d7d881172a06"),
        "base_url": os.getenv("VISION_ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    },
}

# 默认 Provider
DEFAULT_PROVIDER = "aliyun"  # 默认使用阿里大平台

# 向后兼容：提供快捷变量
VISION_API_KEY = VISION_PROVIDERS[DEFAULT_PROVIDER]["api_key"]
VISION_BASE_URL = VISION_PROVIDERS[DEFAULT_PROVIDER]["base_url"]


# 支持的视觉模型配置（基于测试结果）
@dataclass
class VisionModelConfig:
    """视觉模型配置"""
    model_id: str          # API 模型 ID
    display_name: str       # 显示名称
    quality_level: str     # 质量等级
    speed: str             # 速度评级
    recommended: bool      # 是否推荐


VISION_MODELS = {
    # 首选（综合最强，阿里云第一）
    "kimi-k2.6": VisionModelConfig(
        model_id="kimi-k2.6",
        display_name="Kimi K2.6",
        quality_level="B级",
        speed="中",
        recommended=True,
    ),
    # 高质量备选（格式最规范）
    "kimi-k2.5": VisionModelConfig(
        model_id="kimi-k2.5",
        display_name="Kimi K2.5",
        quality_level="B级",
        speed="中",
        recommended=False,
    ),
    # 细节最丰富（但耗时118s）
    "qwen3.6-max-preview": VisionModelConfig(
        model_id="qwen3.6-max-preview",
        display_name="Qwen 3.6 Max Preview",
        quality_level="B级",
        speed="慢",
        recommended=False,
    ),
    # 平衡之选
    "qwen3.6-plus": VisionModelConfig(
        model_id="qwen3.6-plus",
        display_name="Qwen 3.6 Plus",
        quality_level="B级",
        speed="慢",
        recommended=False,
    ),
    # 速度最快（23秒）
    "qwen3.6-flash": VisionModelConfig(
        model_id="qwen3.6-flash",
        display_name="Qwen 3.6 Flash",
        quality_level="B级",
        speed="快",
        recommended=False,
    ),
    # 稳定可用
    "qwen3.5-plus": VisionModelConfig(
        model_id="qwen3.5-plus",
        display_name="Qwen 3.5 Plus",
        quality_level="C级",
        speed="中",
        recommended=False,
    ),
    # 快速筛选
    "qwen3.5-flash": VisionModelConfig(
        model_id="qwen3.5-flash",
        display_name="Qwen 3.5 Flash",
        quality_level="B级",
        speed="快",
        recommended=False,
    ),
}


@dataclass
class VisionResult:
    """视觉分析结果"""
    success: bool
    model: str
    response: str = ""
    usage: Optional[dict] = None
    error: Optional[str] = None
    latency: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0) if self.usage else 0


class VisionModel:
    """
    视觉模型统一封装

    支持多模型切换，默认使用 kimi-k2.6（阿里云平台第一名，73分）
    支持多 Provider 切换（默认使用阿里大平台）
    提供与 glm.py 相似的接口风格

    使用示例：
        # 默认使用阿里大平台 + kimi-k2.6
        model = VisionModel()

        # 指定 Provider
        model = VisionModel(provider="markai")

        # 指定模型
        model = VisionModel(model="kimi-k2.6")

        # 指定 Provider 和模型
        model = VisionModel(provider="aliyun", model="kimi-k2.6")

        # 使用推荐配置
        model = VisionModel.use_recommended("balanced")  # kimi-k2.6
        model = VisionModel.use_recommended("speed")     # qwen3.6-flash

        result = model.analyze_image("test.png", "描述图片内容")
    """

    def __init__(
        self,
        model: str = "kimi-k2.6",
        provider: Literal["markai", "aliyun"] = "aliyun",
        api_key: str = None,
        base_url: str = None,
        timeout: float = 120.0,
    ):
        """
        初始化视觉模型

        Args:
            model: 模型名称，支持 kimi-k2.6/kimi-k2.5/qwen3.6-plus/qwen3.6-flash/qwen3.5-plus/qwen3.5-flash/qwen3.6-max-preview
            provider: Provider 名称，默认 "aliyun"（阿里大平台）
            api_key: API密钥，默认使用对应 provider 的配置
            base_url: API地址，默认使用对应 provider 的配置
            timeout: 超时时间（秒）
        """
        if model not in VISION_MODELS:
            raise ValueError(f"不支持的模型: {model}，支持的模型: {list(VISION_MODELS.keys())}")

        if provider not in VISION_PROVIDERS:
            raise ValueError(f"不支持的 Provider: {provider}，支持的 Provider: {list(VISION_PROVIDERS.keys())}")

        self.model = model
        self.config = VISION_MODELS[model]
        self.provider = provider
        provider_config = VISION_PROVIDERS[provider]
        
        # 优先级：显式参数 > provider 配置 > 环境变量
        self.api_key = api_key or provider_config["api_key"]
        self.base_url = base_url or provider_config["base_url"]
        self.timeout = timeout

    @classmethod
    def use_recommended(cls, scenario: str = "balanced", provider: Literal["markai", "aliyun"] = "aliyun") -> "VisionModel":
        """
        使用推荐配置的模型

        Args:
            scenario: 场景类型
                - "balanced": 平衡模式，默认 kimi-k2.6（阿里云第一）
                - "quality": 质量模式，使用 kimi-k2.5
                - "speed": 速度模式，使用 qwen3.6-flash（最快23s）
                - "deep": 深度分析，使用 qwen3.6-max-preview（细节最丰富）
            provider: Provider 名称，默认使用 DEFAULT_PROVIDER

        Returns:
            VisionModel 实例
        """
        scenarios = {
            "balanced": "kimi-k2.6",
            "quality": "kimi-k2.5",
            "speed": "qwen3.6-flash",
            "deep": "qwen3.6-max-preview",
        }
        model_name = scenarios.get(scenario, "kimi-k2.6")
        return cls(model=model_name, provider=provider)

    def _encode_image(self, image_path: str) -> str:
        """将图片编码为 base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_image(
        self,
        image_path: str,
        prompt: str = None,
        max_tokens: int = 4096,
    ) -> VisionResult:
        """
        分析图片（同步调用）

        Args:
            image_path: 图片路径
            prompt: 提示词（必填）
            max_tokens: 最大 token 数

        Returns:
            VisionResult 对象
        """
        import time
        start_time = time.time()

        try:
            # 编码图片
            image_base64 = self._encode_image(image_path)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.config.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                            {"type": "text", "text": prompt},
                        ]
                    }
                ],
                "max_tokens": max_tokens,
            }

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

                latency = time.time() - start_time

                return VisionResult(
                    success=True,
                    model=self.model,
                    response=result["choices"][0]["message"]["content"],
                    usage=result.get("usage"),
                    latency=latency,
                )

        except Exception as e:
            latency = time.time() - start_time
            return VisionResult(
                success=False,
                model=self.model,
                error=str(e),
                latency=latency,
            )

    async def analyze_image_async(
        self,
        image_path: str,
        prompt: str = None,
        max_tokens: int = 4096,
    ) -> VisionResult:
        """
        分析图片（异步调用）

        Args:
            image_path: 图片路径
            prompt: 提示词（必填）
            max_tokens: 最大 token 数

        Returns:
            VisionResult 对象
        """
        import asyncio
        import time
        start_time = time.time()

        try:
            image_base64 = self._encode_image(image_path)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.config.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                            {"type": "text", "text": prompt},
                        ]
                    }
                ],
                "max_tokens": max_tokens,
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

                latency = time.time() - start_time

                return VisionResult(
                    success=True,
                    model=self.model,
                    response=result["choices"][0]["message"]["content"],
                    usage=result.get("usage"),
                    latency=latency,
                )

        except Exception as e:
            latency = time.time() - start_time
            return VisionResult(
                success=False,
                model=self.model,
                error=str(e),
                latency=latency,
            )

    def compare_models(
        self,
        image_path: str,
        prompt: str = None,
        models: list = None,
    ) -> dict[str, VisionResult]:
        """
        多模型对比分析

        Args:
            image_path: 图片路径
            prompt: 提示词
            models: 要对比的模型列表，默认 ["kimi-k2.6", "kimi-k2.5", "qwen3.6-plus", "qwen3.6-flash"]

        Returns:
            dict[str, VisionResult]，键为模型名称
        """
        if models is None:
            models = ["kimi-k2.6", "kimi-k2.5", "qwen3.6-plus", "qwen3.6-flash"]

        results = {}
        for model_name in models:
            try:
                model = VisionModel(model=model_name, provider=self.provider)
                results[model_name] = model.analyze_image(image_path, prompt)
            except Exception as e:
                results[model_name] = VisionResult(
                    success=False,
                    model=model_name,
                    error=str(e),
                )

        return results

    def __repr__(self) -> str:
        return f"VisionModel(provider={self.provider}, model={self.model}, quality={self.config.quality_level}, speed={self.config.speed})"


# 便捷函数
def analyze_prototype(
    image_path: str,
    model: str = "kimi-k2.6",
    provider: Literal["markai", "aliyun"] = "aliyun",
    prompt: str = None
) -> VisionResult:
    """
    便捷函数：分析原型图

    Args:
        image_path: 原型图路径
        model: 模型名称，默认 kimi-k2.6（阿里云平台第一名）
        provider: Provider 名称，默认使用 DEFAULT_PROVIDER
        prompt: 提示词

    Returns:
        VisionResult 对象
    """
    model_instance = VisionModel(model=model, provider=provider)
    return model_instance.analyze_image(image_path, prompt)


def compare_prototype_models(
    image_path: str, 
    models: list = None,
    provider: Literal["markai", "aliyun"] = "aliyun"
) -> dict[str, VisionResult]:
    """
    便捷函数：多模型对比分析原型图

    Args:
        image_path: 原型图路径
        models: 要对比的模型列表
        provider: Provider 名称，默认使用 DEFAULT_PROVIDER

    Returns:
        dict[str, VisionResult]
    """
    model = VisionModel(provider=provider)
    return model.compare_models(image_path, models=models)


def get_provider_info(provider: Literal["markai", "aliyun"] = "aliyun") -> dict:
    """
    获取 Provider 信息

    Args:
        provider: Provider 名称，默认使用 DEFAULT_PROVIDER

    Returns:
        dict，包含 provider 的配置信息
    """
    if provider is None:
        provider = DEFAULT_PROVIDER
    return VISION_PROVIDERS.get(provider, {})


def list_providers() -> list[str]:
    """
    列出所有可用的 Provider

    Returns:
        Provider 名称列表
    """
    return list(VISION_PROVIDERS.keys())
