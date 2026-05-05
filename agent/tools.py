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
            collection_name="design_references",
            use_remote=False,
            persist_directory="./chroma_data",
        )
    return _standards_store


def get_prototype_store() -> PrototypeStore:
    global _prototype_store
    if _prototype_store is None:
        _prototype_store = PrototypeStore()
    return _prototype_store


def get_prd_store() -> ChromaStore:
    """获取 PRD 文档存储"""
    return ChromaStore(
        collection_name="prd_documents",
        persist_directory="./chroma_data",
        use_remote=False,
    )


@tool
def retrieve_prd_chunks(query: str, document_id: str = None, k: int = 10) -> str:
    """
    从 PRD 文档库检索相关内容片段

    Args:
        query: 查询文本
        document_id: 可选，限定检索的文档 ID
        k: 返回的 chunk 数量（默认 10）

    Returns:
        相关文档片段列表
    """
    from vec_db.embeddings.bge_m3_embedding import BgeM3Embeddings

    store = get_prd_store()
    embedder = BgeM3Embeddings()
    query_embedding = embedder.embed_query(query)

    results = store.similarity_search(query_embedding, k=k)

    if not results:
        return "未找到相关的 PRD 文档内容"

    formatted = []
    for doc, score in results:
        meta = doc.metadata
        # 如果指定了 document_id，进行过滤
        if document_id and meta.get("document_id") != document_id:
            continue

        heading = meta.get("heading", "未命名")
        chunk_idx = meta.get("chunk_index", "?")
        header_path = meta.get("header_path", "")

        formatted.append(
            f"--- [{heading}] (Chunk #{chunk_idx}, 相似度: {score:.2f}) ---\n"
            f"路径: {header_path}\n"
            f"{doc.page_content[:500]}"
            f"{'...' if len(doc.page_content) > 500 else ''}"
        )

    if not formatted:
        return "未找到与查询相关的 PRD 文档内容"

    return "\n\n".join(formatted[:k])


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
        retrieve_prd_chunks,
        analyze_prd,
        analyze_prototype,
        generate_report,
        store_prototype_description,
    ]
