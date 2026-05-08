# PRD & 原型图审查系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 构建一个基于 LangGraph + LangChain 的 PRD 和原型图审查 Agent，使用 MiniMax-M2.7 作为 Agent 主体，GLM-4.6V-Flash 处理视觉分析

**架构：** 单 Agent 模式（ReAct），Tools 绑定到 Agent，包含 RAG 检索工具和 LLM 分析工具。上下文管理采用 RAG 检索 + 按需获取策略

**技术栈：** Python 3.12, uv, langchain, langgraph, chromadb, MiniMax-M2.7, GLM-4.6V-Flash

---

## 文件结构

```
dr-2/
├── agent/
│   ├── __init__.py
│   ├── review_agent.py      # Agent 创建和执行入口
│   ├── tools.py             # Tools 定义（5个 tool）
│   ├── prompts.py           # System prompts
│   └── schemas.py           # 数据模型（Pydantic）
├── vec_db/
│   └── store/
│       └── prototype_store.py  # prototype_descriptions collection 操作
├── models/
│   ├── __init__.py
│   ├── minimax.py           # MiniMax 模型封装
│   └── glm.py               # GLM 模型封装
├── pyproject.toml           # 添加依赖
└── main.py                  # 入口
```

---

## Task 1: 依赖安装

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 添加 langchain 相关依赖**

修改 `pyproject.toml`，添加：

```toml
[project]
dependencies = [
    "asyncpg>=0.31.0",
    "chromadb[postgresql]>=0.5.0",
    "langchain>=1.2.15",
    "langchain-community>=0.4.1",
    "langchain-core>=1.3.2",
    "psycopg2-binary>=2.9.12",
    "sentence-transformers>=5.4.1",
    # 新增
    "langgraph>=0.0.20",
    "langchain-openai>=0.1.0",
    "openai>=1.12.0",
    "pydantic>=2.0",
]
```

- [ ] **Step 2: 安装依赖**

```bash
uv sync
```

- [ ] **Step 3: 提交**

```bash
git add pyproject.toml
git commit -m "feat: 添加 langgraph, langchain-openai, openai, pydantic 依赖"
```

---

## Task 2: 模型封装

**Files:**
- Create: `models/__init__.py`
- Create: `models/minimax.py`
- Create: `models/glm.py`

- [ ] **Step 1: 创建 models/__init__.py**

```python
from models.minimax import MiniMaxModel
from models.glm import GLMModel

__all__ = ["MiniMaxModel", "GLMModel"]
```

- [ ] **Step 2: 创建 models/minimax.py**

```python
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
```

- [ ] **Step 3: 创建 models/glm.py**

```python
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
```

- [ ] **Step 4: 提交**

```bash
git add models/
git commit -m "feat: 添加 MiniMax 和 GLM 模型封装"
```

---

## Task 3: 数据模型

**Files:**
- Create: `agent/schemas.py`

- [ ] **Step 1: 创建 agent/schemas.py**

```python
"""Agent 数据模型"""

from typing import Optional
from pydantic import BaseModel, Field

class PrototypeDescription(BaseModel):
    """原型图描述"""
    id: str = Field(description="唯一标识 (UUID)，内部使用")
    name: str = Field(description="用户可读名称，如'登录页'、'主页导航'")
    document_id: str = Field(description="关联的 PRD 文档 ID")
    page_name: str = Field(description="页面名称")
    layout: str = Field(description="布局描述")
    components: list[str] = Field(description="组件列表")
    interactions: str = Field(description="交互描述")
    query_text: str = Field(description="用于检索的文本（聚合 name + page_name + layout）")

class ReviewResult(BaseModel):
    """审查结果"""
    prd_analysis: str = Field(description="PRD 分析结果")
    prototype_analysis: str = Field(description="原型图分析结果")
    compliance_check: str = Field(description="合规性检查结果")
    issues: list[str] = Field(description="问题列表")
    suggestions: list[str] = Field(description="建议列表")

class ReviewReport(BaseModel):
    """审查报告"""
    title: str = Field(description="报告标题")
    document_name: str = Field(description="被审查的文档名")
    prototype_name: Optional[str] = Field(description="被审查的原型图名")
    review_result: ReviewResult = Field(description="审查结果")
    standards_referenced: list[str] = Field(description="引用的设计标准")
```

- [ ] **Step 2: 提交**

```bash
git add agent/schemas.py
git commit -m "feat: 添加 Agent 数据模型 (Pydantic)"
```

---

## Task 4: 原型描述存储

**Files:**
- Create: `vec_db/store/prototype_store.py`

- [ ] **Step 1: 创建 vec_db/store/prototype_store.py**

```python
"""原型图描述存储 - 使用 ChromaDB"""

import uuid
from typing import Optional

from vec_db.store.chroma_store import ChromaStore
from agent.schemas import PrototypeDescription


class PrototypeStore:
    """原型图描述存储管理"""

    def __init__(
        self,
        collection_name: str = "prototype_descriptions",
        persist_directory: str = "./chroma_data",
        use_remote: bool = False,
    ):
        self.chroma_store = ChromaStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            use_remote=use_remote,
        )

    def store(
        self,
        description: PrototypeDescription,
        embedding: list[float],
    ) -> str:
        """存储原型描述（防重复：基于 document_id + page_name）"""
        from langchain_core.documents import Document

        # 检查是否已存在（基于 document_id + page_name）
        existing = self.chroma_store._collection.get(
            where={
                "document_id": description.document_id,
                "page_name": description.page_name
            }
        )

        if existing and existing.get("ids"):
            # 已存在，更新而非重复插入
            doc_id = existing["ids"][0]
            self.chroma_store._collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[description.query_text],
                metadatas=[{
                    "id": doc_id,
                    "name": description.name,
                    "document_id": description.document_id,
                    "page_name": description.page_name,
                    "layout": description.layout,
                    "components": ",".join(description.components),
                    "interactions": description.interactions,
                    "query_text": description.query_text,
                }]
            )
            return doc_id

        # 新增
        doc_id = description.id or str(uuid.uuid4())

        doc = Document(
            page_content=description.query_text,
            metadata={
                "id": doc_id,
                "name": description.name,
                "document_id": description.document_id,
                "page_name": description.page_name,
                "layout": description.layout,
                "components": ",".join(description.components),
                "interactions": description.interactions,
                "query_text": description.query_text,
            }
        )

        # 使用 upsert 存储
        self.chroma_store._collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc.page_content],
            metadatas=[doc.metadata],
        )

        return doc_id

    def retrieve(
        self,
        query_embedding: list[float],
        k: int = 5,
        document_id: Optional[str] = None,
    ) -> list[PrototypeDescription]:
        """检索原型描述（通过 name 或 query_text 语义检索）"""
        results = self.chroma_store.similarity_search(
            embedding=query_embedding,
            k=k,
        )

        descriptions = []
        for doc, score in results:
            meta = doc.metadata
            if document_id and meta.get("document_id") != document_id:
                continue

            descriptions.append(PrototypeDescription(
                id=meta.get("id", ""),
                name=meta.get("name", ""),
                document_id=meta.get("document_id", ""),
                page_name=meta.get("page_name", ""),
                layout=meta.get("layout", ""),
                components=meta.get("components", "").split(",") if meta.get("components") else [],
                interactions=meta.get("interactions", ""),
                query_text=meta.get("query_text", ""),
            ))

        return descriptions
```

- [ ] **Step 2: 提交**

```bash
git add vec_db/store/prototype_store.py
git commit -m "feat: 添加 PrototypeStore 用于存储原型图描述"
```

---

## Task 5: System Prompts

**Files:**
- Create: `agent/prompts.py`

- [ ] **Step 1: 创建 agent/prompts.py**

```python
"""Agent System Prompts"""

REVIEW_AGENT_PROMPT = """你是一个专业的 PRD 和原型图审查专家。

## 你的角色
用户将上传 PRD 文档和原型图，你需要：
1. 分析 PRD 的完整性和一致性
2. 分析原型图的布局和元素
3. 对照设计标准检查合规性
4. 输出结构化的审查报告

## 可用工具
- retrieve_standards: 检索设计标准（从 ChromaDB design_reference collection）
- retrieve_prototypes: 检索历史原型描述（从 ChromaDB prototype_descriptions collection）
- analyze_prd: 分析 PRD 文档内容
- analyze_prototype: 分析原型图（需要传入图片路径）
- generate_report: 生成结构化审查报告

## 工作流程
1. 首先使用 retrieve_standards 检索相关设计标准
2. 使用 analyze_prd 分析 PRD 文档
3. 使用 analyze_prototype 分析原型图（传入图片路径）
4. 如有需要，使用 retrieve_prototypes 检索历史原型描述
5. 使用 generate_report 生成最终报告

## 输出格式
请按以下格式输出：

## 审查结果

### PRD 分析
[PRD 的完整性、一致性、术语规范性分析]

### 原型图分析
[原型图的布局、组件、交互元素分析]

### 合规性检查
[与设计标准的对照检查结果]

### 问题列表
1. [问题1]
2. [问题2]
...

### 建议
1. [建议1]
2. [建议2]
...
"""

RETRIEVE_STANDARDS_PROMPT = """从设计标准知识库中检索与以下查询相关的内容：

{query}

返回相关标准片段列表。"""

RETRIEVE_PROTOTYPES_PROMPT = """从原型描述库中检索与以下查询相关的内容：

{query}

返回相关的原型描述列表。"""

ANALYZE_PRD_PROMPT = """分析以下 PRD 文档：

{content}

请提取：
1. 文档结构
2. 主要功能模块
3. 潜在问题（歧义、遗漏、不一致）
4. 术语规范性

以结构化格式返回分析结果。"""

ANALYZE_PROTOTYPE_PROMPT = """分析这张原型图：

请提取：
1. 页面名称/标题
2. 布局结构
3. 组件列表（按钮、表单、列表、导航等）
4. 交互元素（跳转、弹窗、状态变化等）
5. 是否符合常见设计规范

以结构化格式返回分析结果。"""

GENERATE_REPORT_PROMPT = """根据以下审查结果生成最终报告：

PRD 分析：{prd_analysis}
原型图分析：{prototype_analysis}
合规性检查：{compliance_check}
问题列表：{issues}
建议：{suggestions}

生成完整的 Markdown 格式审查报告。"""
```

- [ ] **Step 2: 提交**

```bash
git add agent/prompts.py
git commit -m "feat: 添加 ReviewAgent System Prompt"
```

---

## Task 6: Tools 定义

**Files:**
- Create: `agent/tools.py`

- [ ] **Step 1: 创建 agent/tools.py**

```python
"""Agent Tools 定义"""

from typing import Annotated
from langchain_core.tools import tool

from models.minimax import MiniMaxModel
from models.glm import GLMModel
from vec_db.store.chroma_store import ChromaStore
from vec_db.store.prototype_store import PrototypeStore
from agent.prompts import (
    RETRIEVE_STANDARDS_PROMPT,
    RETRIEVE_PROTOTYPES_PROMPT,
    ANALYZE_PRD_PROMPT,
    ANALYZE_PROTOTYPE_PROMPT,
    GENERATE_REPORT_PROMPT,
)
from agent.schemas import PrototypeDescription, ReviewResult, ReviewReport


# 全局模型实例
_minimax_model = None
_glm_model = None
_standards_store = None
_prototype_store = None


def get_minimax_model() -> MiniMaxModel:
    global _minimax_model
    if _minimax_model is None:
        _minimax_model = MiniMaxModel()
    return _minimax_model


def get_glm_model() -> GLMModel:
    global _glm_model
    if _glm_model is None:
        _glm_model = GLMModel()
    return _glm_model


def get_standards_store() -> ChromaStore:
    global _standards_store
    if _standards_store is None:
        _standards_store = ChromaStore(
            collection_name="design_reference",
            use_remote=False,
            persist_directory="./chroma_data",
        )
    return _standards_store


def get_prototype_store() -> PrototypeStore:
    global _prototype_store
    if _prototype_store is None:
        _prototype_store = PrototypeStore()
    return _prototype_store


@tool
def retrieve_standards(query: str) -> str:
    """从设计标准知识库检索相关内容"""
    store = get_standards_store()

    # 获取 query 的 embedding
    from vec_db.embeddings.bge_m3_embedding import BgeM3Embeddings
    embedder = BgeM3Embeddings()
    query_embedding = embedder.embed_query(query)

    # 检索
    results = store.similarity_search(query_embedding, k=5)

    if not results:
        return "未找到相关的设计标准"

    # 格式化结果
    formatted = []
    for doc, score in results:
        formatted.append(f"- [{doc.metadata.get('heading', '标准片段')}] (相似度: {score:.2f})\n  {doc.page_content[:200]}...")

    return "\n\n".join(formatted)


@tool
def retrieve_prototypes(query: str, document_id: str = None) -> str:
    """从原型描述库检索相关内容"""
    store = get_prototype_store()

    from vec_db.embeddings.bge_m3_embedding import BgeM3Embeddings
    embedder = BgeM3Embeddings()
    query_embedding = embedder.embed_query(query)

    results = store.retrieve(query_embedding, k=5, document_id=document_id)

    if not results:
        return "未找到相关的原型描述"

    formatted = []
    for desc in results:
        formatted.append(f"- 页面: {desc.page_name}\n  布局: {desc.layout}\n  组件: {', '.join(desc.components)}\n  交互: {desc.interactions}")

    return "\n\n".join(formatted)


@tool
def analyze_prd(content: str) -> str:
    """分析 PRD 文档内容"""
    model = get_minimax_model()

    prompt = ANALYZE_PRD_PROMPT.format(content=content[:8000])  # 限制长度

    response = model.invoke([
        {"role": "system", "content": "你是一个专业的 PRD 分析师。"},
        {"role": "user", "content": prompt},
    ])

    return response.content if hasattr(response, 'content') else str(response)


@tool
def analyze_prototype(image_path: str) -> str:
    """分析原型图（需要图片路径）"""
    glm = get_glm_model()

    prompt = """分析这张原型图，请提取：
1. 页面名称/标题
2. 布局结构（头部、内容区、底部等）
3. 组件列表（按钮、表单、列表、导航等）
4. 交互元素（跳转、弹窗、状态变化等）
5. 视觉规范（间距、配色、对齐等）

请以结构化格式返回。"""

    try:
        result = glm.analyze_image(image_path, prompt)
        return result
    except Exception as e:
        return f"原型图分析失败: {str(e)}"


@tool
def generate_report(
    prd_analysis: str,
    prototype_analysis: str,
    compliance_check: str,
    issues: list[str],
    suggestions: list[str],
    document_name: str = "",
    prototype_name: str = "",
    standards_referenced: list[str] = None,
) -> str:
    """生成结构化审查报告"""
    if standards_referenced is None:
        standards_referenced = []

    model = get_minimax_model()

    prompt = GENERATE_REPORT_PROMPT.format(
        prd_analysis=prd_analysis,
        prototype_analysis=prototype_analysis,
        compliance_check=compliance_check,
        issues="\n".join(f"{i+1}. {issue}" for i, issue in enumerate(issues)),
        suggestions="\n".join(f"{i+1}. {sug}" for i, sug in enumerate(suggestions)),
    )

    response = model.invoke([
        {"role": "system", "content": "你是一个专业的技术写作专家。"},
        {"role": "user", "content": prompt},
    ])

    return response.content if hasattr(response, 'content') else str(response)


@tool
def store_prototype_description(
    name: str,
    document_id: str,
    page_name: str,
    layout: str,
    components: list[str],
    interactions: str,
) -> str:
    """存储原型图描述到向量数据库（防重复：基于 document_id + page_name）"""
    from vec_db.embeddings.bge_m3_embedding import BgeM3Embeddings

    prototype_store = get_prototype_store()
    embedder = BgeM3Embeddings()

    # 生成 query_text 用于检索
    query_text = f"{name} {page_name} {layout} {' '.join(components)}"

    # 生成 embedding
    embedding = embedder.embed_query(query_text)

    # 创建描述对象
    description = PrototypeDescription(
        id="",
        name=name,
        document_id=document_id,
        page_name=page_name,
        layout=layout,
        components=components,
        interactions=interactions,
        query_text=query_text,
    )

    # 存储
    doc_id = prototype_store.store(description, embedding)

    return f"原型描述已存储，ID: {doc_id}"


def get_all_tools():
    """获取所有工具列表"""
    return [
        retrieve_standards,
        retrieve_prototypes,
        analyze_prd,
        analyze_prototype,
        generate_report,
        store_prototype_description,
    ]
```

- [ ] **Step 2: 提交**

```bash
git add agent/tools.py
git commit -m "feat: 添加 Agent Tools (6个工具)"
```

---

## Task 7: ReviewAgent 创建

**Files:**
- Create: `agent/review_agent.py`
- Create: `agent/__init__.py`

- [ ] **Step 1: 创建 agent/__init__.py**

```python
from agent.review_agent import ReviewAgent

__all__ = ["ReviewAgent"]
```

- [ ] **Step 2: 创建 agent/review_agent.py**

```python
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
```

- [ ] **Step 3: 提交**

```bash
git add agent/review_agent.py agent/__init__.py
git commit -m "feat: 添加 ReviewAgent (LangGraph ReAct)"
```

---

## Task 8: 入口文件

**Files:**
- Modify: `main.py`

- [ ] **Step 1: 修改 main.py**

```python
"""PRD & 原型图审查系统入口"""

from agent.review_agent import create_review_agent


def main():
    print("=" * 60)
    print("PRD & 原型图审查系统")
    print("=" * 60)
    print()

    # 创建 Agent
    agent = create_review_agent()

    print("Agent 已就绪，请输入审查请求...")
    print("示例：")
    print("  请审查 PRD 文档 at ref-doc/产品设计标准文档V2.０--25年持续更新.md")
    print("  和原型图 at tests/temp/prototype.png")
    print()

    while True:
        try:
            user_input = input("> ")

            if user_input.lower() in ["exit", "quit", "q"]:
                print("再见！")
                break

            if not user_input.strip():
                continue

            # 调用 Agent
            result = agent.invoke(user_input)

            # 输出结果
            print("\n--- 审查结果 ---\n")
            for message in result["messages"]:
                if hasattr(message, "content") and message.content:
                    print(message.content)

            print("\n" + "=" * 60)

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add main.py
git commit -m "feat: 添加系统入口 main.py"
```

---

## Task 9: 端到端测试

**Files:**
- Create: `tests/test_review_agent.py`

- [ ] **Step 1: 创建测试文件**

```python
"""ReviewAgent 端到端测试"""

import pytest
from agent.review_agent import create_review_agent
from agent.tools import retrieve_standards, analyze_prd


def test_agent_creation():
    """测试 Agent 创建"""
    agent = create_review_agent()
    assert agent is not None
    assert agent.model is not None


def test_retrieve_standards():
    """测试设计标准检索工具"""
    result = retrieve_standards.invoke("交互设计规范")
    assert isinstance(result, str)


def test_analyze_prd():
    """测试 PRD 分析工具"""
    test_prd = """
    # 产品需求文档

    ## 功能需求

    ### 用户管理
    1. 用户可以注册账号
    2. 用户可以登录
    3. 用户可以修改密码

    ### 权限管理
    - 管理员可以查看所有用户
    - 普通用户只能查看自己的信息
    """

    result = analyze_prd.invoke(test_prd)
    assert isinstance(result, str)
    assert len(result) > 0


def test_agent_invoke():
    """测试 Agent 调用（需要 API key）"""
    agent = create_review_agent()

    # 简单测试
    result = agent.invoke("你好，请介绍一下你自己")

    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: 运行测试**

```bash
cd C:/AA_GSE_DP/DR-2
python -m pytest tests/test_review_agent.py -v
```

- [ ] **Step 3: 提交**

```bash
git add tests/test_review_agent.py
git commit -m "test: 添加 ReviewAgent 端到端测试"
```

---

## 验证清单

在完成所有 Task 后，验证以下功能：

- [ ] `uv sync` 成功安装所有依赖
- [ ] `from models import MiniMaxModel, GLMModel` 导入成功
- [ ] `from agent.tools import get_all_tools` 导入成功，6个工具已注册
- [ ] `from agent.review_agent import ReviewAgent` 导入成功
- [ ] `python main.py` 可以启动 Agent
- [ ] `retrieve_standards` 工具可以连接 ChromaDB 检索
- [ ] `analyze_prd` 工具可以调用 MiniMax 分析文本
- [ ] `analyze_prototype` 工具可以调用 GLM 分析图片
- [ ] 测试文件可以正常运行

---

*计划完成时间：2026-04-30*
*预计 Task 数量：9 个*
*预计完成时间：1-2 小时（取决于环境配置）*