"""Agent Tools 定义"""

from typing import Annotated
from langchain_core.tools import tool

from models.minimax import MiniMaxModel
from models.vision import VisionModel
from vec_db.store.chroma_store import ChromaStore
from vec_db.store.prototype_store import PrototypeStore
from agent.prompts import (
    RETRIEVE_STANDARDS_PROMPT,
    RETRIEVE_PROTOTYPES_PROMPT,
    ANALYZE_PRD_PROMPT,
    ANALYZE_PROTOTYPE_PROMPT,
    GENERATE_REPORT_PROMPT,
    EXTRACT_STANDARDS_RULES_PROMPT,
    GENERATE_COMPARISON_REPORT_PROMPT,
)
from agent.schemas import PrototypeDescription, ReviewResult, ReviewReport


# 全局模型实例
_minimax_model = None
_vision_model = None
_standards_store = None
_prototype_store = None


def get_minimax_model() -> MiniMaxModel:
    global _minimax_model
    if _minimax_model is None:
        _minimax_model = MiniMaxModel()
    return _minimax_model


def get_vision_model() -> VisionModel:
    global _vision_model
    if _vision_model is None:
        _vision_model = VisionModel()  # 默认 qwen3.6-plus
    return _vision_model


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

    # 构建过滤条件
    filter_cond = {"document_id": document_id} if document_id else None
    # 多取一些，因为过滤在数据库层做了
    results = store.similarity_search(query_embedding, k=k, filter=filter_cond)

    if not results:
        return "未找到相关的 PRD 文档内容" if not document_id else f"未找到文档 {document_id} 相关的内容"

    formatted = []
    for doc, score in results:
        meta = doc.metadata
        heading = meta.get("heading", "未命名")
        chunk_idx = meta.get("chunk_index", "?")
        header_path = meta.get("header_path", "")

        formatted.append(
            f"--- [{heading}] (Chunk #{chunk_idx}, 相似度: {score:.2f}) ---\n"
            f"路径: {header_path}\n"
            f"{doc.page_content[:500]}"
            f"{'...' if len(doc.page_content) > 500 else ''}"
        )

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
def analyze_prototype(image_path: str) -> dict:
    """
    分析原型图（需要图片路径）

    Returns:
        dict: 包含 status, page_name, layout, components, interactions, error 等字段
    """
    import os

    # 检查文件是否存在
    if not os.path.exists(image_path):
        # 尝试多种路径组合
        possible_paths = [
            image_path,
            os.path.join(os.getcwd(), image_path),
            os.path.join(os.getcwd(), "prd-test", os.path.basename(image_path)),
            os.path.join(os.getcwd(), "prd-test", image_path),
        ]

        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break

        if not found_path:
            return {
                "status": "error",
                "error": f"原型图文件不存在: {image_path}",
                "tried_paths": possible_paths,
            }
        image_path = found_path

    vision = get_vision_model()

    try:
        result = vision.analyze_image(image_path, ANALYZE_PROTOTYPE_PROMPT)
        if result.success:
            return {
                "status": "success",
                "raw_result": result.response,
                "usage": result.usage,
                "latency": result.latency,
                "image_path": image_path,
            }
        else:
            return {
                "status": "error",
                "error": result.error,
                "image_path": image_path,
            }
    except Exception as e:
        return {
            "status": "error",
            "error": f"原型图分析失败: {str(e)}",
            "image_path": image_path,
        }


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

    # 处理 LangChain AIMessage 对象的 content（可能是字符串或列表）
    content = response.content if hasattr(response, 'content') else str(response)
    if isinstance(content, list):
        # 提取纯文本，过滤掉思考内容
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "thinking":
                    # 跳过思考内容
                    pass
            elif isinstance(block, str):
                text_parts.append(block)
        content = "\n".join(text_parts)
    elif isinstance(content, str):
        # 过滤掉思考内容标记
        import re
        # 移除 <think>...</think> 模式
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        content = content.strip()
    return content


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


@tool
def enrich_prototype(
    prototype_id: str,
    layout: str,
    components: list[str],
    interactions: str,
    page_type: str = None,
    fidelity_level: str = None,
    module_path: str = None,
    layout_type: str = None,
    min_resolution: str = None,
    primary_color: str = None,
    font_hierarchy: dict = None,
    component_spacing: str = None,
    button_types: list[str] = None,
    form_fields_per_row: int = None,
    table_column_count: int = None,
    table_page_size: int = None,
    popup_types: list[str] = None,
    filter_default_fields: int = None,
    filter_total_fields: int = None,
    stat_card_count: int = None,
    component_library: str = None,
    currency_format: str = None,
    date_format: str = None,
    validation_type: str = None,
    validation_rules: dict = None,
    full_content: str = None,
    extracted_specs: dict = None,
    unmeasurable_specs: list[str] = None,
) -> str:
    """
    在 review 阶段分析完成后，用分析结果丰富原型图存储。

    该工具在 analyze_prototype 分析完原型图后调用，用于：
    1. 将分析结果（layout, components, interactions 等）存入 PrototypeStore
    2. 更新 SQLite 中 session.prototypes 的丰富字段

    Args:
        prototype_id: 原型图 ID
        layout: 布局描述
        components: 组件列表
        interactions: 交互描述
        page_type: 页面类型 (列表页/详情页/表单页/统计页等)
        fidelity_level: 保真度等级 (低/中/高)
        module_path: 所属模块/导航路径
        layout_type: 整体布局形式 (左右/上下等)
        min_resolution: 最低适应分辨率
        primary_color: 主色调
        font_hierarchy: 字体层级 {"title": "22px", "body": "16px"}
        component_spacing: 组件间距规范
        button_types: 按钮类型列表
        form_fields_per_row: 单行字段数量
        table_column_count: 表格列数
        table_page_size: 分页设置/每页条数
        popup_types: 弹窗类型列表
        filter_default_fields: 默认展示字段数
        filter_total_fields: 总字段数
        stat_card_count: 统计卡片数量
        component_library: 使用的组件库 (AntDesign等)
        currency_format: 金额格式
        date_format: 日期格式
        validation_type: 验证方式 (即时/提交)
        validation_rules: 验证规则详情
        full_content: 大模型输出的完整原始分析文本
        extracted_specs: 提取的规格值 JSON
        unmeasurable_specs: 无法从原型判断的规格列表
    """
    from vec_db.store.prototype_store import get_prototype_store
    import json

    prototype_store = get_prototype_store()

    # 构建 enriched_data
    enriched_data = {
        "layout": layout,
        "components": components,
        "interactions": interactions,
    }

    # 添加所有可选字段
    optional_fields = {
        "page_type": page_type,
        "fidelity_level": fidelity_level,
        "module_path": module_path,
        "layout_type": layout_type,
        "min_resolution": min_resolution,
        "primary_color": primary_color,
        "font_hierarchy": font_hierarchy,
        "component_spacing": component_spacing,
        "button_types": button_types,
        "form_fields_per_row": form_fields_per_row,
        "table_column_count": table_column_count,
        "table_page_size": table_page_size,
        "popup_types": popup_types,
        "filter_default_fields": filter_default_fields,
        "filter_total_fields": filter_total_fields,
        "stat_card_count": stat_card_count,
        "component_library": component_library,
        "currency_format": currency_format,
        "date_format": date_format,
        "validation_type": validation_type,
        "validation_rules": validation_rules,
        "full_content": full_content,
        "extracted_specs": extracted_specs,
        "unmeasurable_specs": unmeasurable_specs,
    }

    for key, value in optional_fields.items():
        if value is not None:
            enriched_data[key] = value

    # 更新 PrototypeStore (ChromaDB)
    success = prototype_store.update_enriched(prototype_id, enriched_data)

    if success:
        return f"原型 {prototype_id} 已丰富存储"
    else:
        return f"原型 {prototype_id} 丰富存储失败"


@tool
def extract_prd_specs(prd_content: str) -> str:
    """
    从 PRD 文档内容中提取可量化的规格值，生成结构化 JSON。

    输入: PRD 文档的完整文本内容
    输出: JSON 格式的规格值字典，包含维度/具体属性、值、上下文

    用于与原型图实现值、设计标准规范进行三栏对比报告。
    """
    model = get_minimax_model()
    prompt = ANALYZE_PRD_PROMPT.format(prd_content=prd_content)

    messages = [
        {"role": "system", "content": "你是一个专业的 PRD 审查专家，负责提取文档中的可量化规格值。"},
        {"role": "user", "content": prompt},
    ]

    response = model.invoke(messages)
    return response.content if hasattr(response, 'content') else str(response)


@tool
def extract_prototype_specs(prototype_analysis: str) -> str:
    """
    从原型图分析结果中提取可量化的规格值，生成结构化 JSON。

    输入: analyze_prototype 工具返回的分析结果文本
    输出: JSON 格式的原型规格字典，包含维度/具体属性、值、可信度、上下文

    用于与 PRD 规格值、设计标准规范进行三栏对比报告。
    """
    model = get_minimax_model()

    messages = [
        {"role": "system", "content": "你是一个专业的原型图审查专家，负责从原型图分析结果中提取可量化的规格值。"},
        {"role": "user", "content": f"从以下原型图分析结果中提取规格值（第九节的 JSON 部分）：\n\n{prototype_analysis}"},
    ]

    response = model.invoke(messages)
    return response.content if hasattr(response, 'content') else str(response)


@tool
def extract_standard_rules(standards_content: str) -> str:
    """
    从设计标准文档片段中提取可量化的规范值，生成结构化 JSON。

    输入: retrieve_standards 工具返回的标准文档片段
    输出: JSON 格式的标准规范字典，包含维度/具体属性、值、范围、来源

    用于与 PRD 规格值、原型图实现值进行三栏对比报告。
    """
    model = get_minimax_model()
    prompt = EXTRACT_STANDARDS_RULES_PROMPT.format(standards_content=standards_content)

    messages = [
        {"role": "system", "content": "你是一个专业的设计标准审查专家，负责从标准文档中提取可量化的规范值。"},
        {"role": "user", "content": prompt},
    ]

    response = model.invoke(messages)
    return response.content if hasattr(response, 'content') else str(response)


@tool
def generate_comparison_report(
    prd_specs: str,
    prototype_specs: str,
    standard_rules: str,
    document_name: str = "",
    prototype_name: str = "",
) -> str:
    """
    生成对比式合规性审查报告。

    输入三个 JSON 格式的规格字典，生成三栏对比 Markdown 报告：
    - PRD 规格值
    - 原型图实现值
    - 设计标准规范

    自动匹配语义等价的维度，并判断：
    - ✅ 完全一致
    - ⚠️ 超出标准范围
    - ❌ 三者均不一致
    - — 未提及
    """
    model = get_minimax_model()

    prompt = GENERATE_COMPARISON_REPORT_PROMPT.format(
        prd_specs=prd_specs,
        prototype_specs=prototype_specs,
        standard_rules=standard_rules,
    )

    messages = [
        {"role": "system", "content": "你是一个专业的设计合规性审查报告生成专家，负责生成对比式 Markdown 报告。"},
        {"role": "user", "content": prompt},
    ]

    response = model.invoke(messages)
    content = response.content if hasattr(response, 'content') else str(response)

    # 在报告头部添加文档信息
    header = f"> PRD 文档：{document_name}\n> 原型图：{prototype_name}\n\n"
    if content.startswith("##"):
        # 找到第一个 ## 标题的位置，在其后插入 header
        first_hash = content.find("##")
        if first_hash != -1:
            content = content[:first_hash + 2] + " 合规性审查报告\n\n" + header + content[first_hash + 2:]

    return content


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
        extract_prd_specs,
        extract_prototype_specs,
        extract_standard_rules,
        enrich_prototype,
        generate_comparison_report,
    ]
