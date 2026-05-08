"""Agent 数据模型"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class PrototypeDescription(BaseModel):
    """原型图描述"""
    id: str = Field(description="唯一标识 (UUID)，内部使用")
    name: str = Field(description="用户可读名称，如'登录页'、'主页导航'")
    document_id: Optional[str] = Field(default=None, description="关联的 PRD 文档 ID（可为空，支持后续绑定）")
    page_name: str = Field(description="页面名称")
    layout: str = Field(default="", description="布局描述")
    components: list[str] = Field(default_factory=list, description="组件列表")
    interactions: str = Field(default="", description="交互描述")
    query_text: str = Field(description="用于检索的文本（聚合 name + page_name + layout）")
    image_path: str = Field(description="原型图文件路径")

    # === NEW: Page metadata ===
    page_type: Optional[str] = Field(default=None, description="页面类型: 列表页/详情页/表单页/统计页等")
    fidelity_level: Optional[str] = Field(default=None, description="保真度等级: 低/中/高")
    module_path: Optional[str] = Field(default=None, description="所属模块/导航路径")

    # === NEW: Layout details ===
    layout_type: Optional[str] = Field(default=None, description="整体布局形式: 左右/上下等")
    min_resolution: Optional[str] = Field(default=None, description="最低适应分辨率")

    # === NEW: Visual elements ===
    primary_color: Optional[str] = Field(default=None, description="主色调")
    font_hierarchy: Optional[dict] = Field(default=None, description="字体层级: {title: size, body: size, auxiliary: size}")
    component_spacing: Optional[str] = Field(default=None, description="组件间距规范")

    # === NEW: Component details ===
    button_types: Optional[list[str]] = Field(default=None, description="按钮类型列表")
    form_fields_per_row: Optional[int] = Field(default=None, description="单行字段数量")
    table_column_count: Optional[int] = Field(default=None, description="表格列数")
    table_page_size: Optional[int] = Field(default=None, description="分页设置/每页条数")
    popup_types: Optional[list[str]] = Field(default=None, description="弹窗类型列表")
    filter_default_fields: Optional[int] = Field(default=None, description="默认展示字段数")
    filter_total_fields: Optional[int] = Field(default=None, description="总字段数")
    stat_card_count: Optional[int] = Field(default=None, description="统计卡片数量")
    component_library: Optional[str] = Field(default=None, description="使用的组件库: AntDesign等")

    # === NEW: Data format ===
    currency_format: Optional[str] = Field(default=None, description="金额格式")
    date_format: Optional[str] = Field(default=None, description="日期格式")

    # === NEW: Validation ===
    validation_type: Optional[str] = Field(default=None, description="验证方式: 即时/提交")
    validation_rules: Optional[dict] = Field(default=None, description="验证规则详情")

    # === NEW: Analysis results ===
    full_content: Optional[str] = Field(default=None, description="大模型输出的完整原始分析文本")
    extracted_specs: Optional[dict] = Field(default=None, description="提取的规格值 JSON: {dimension: {value, confidence, context}}")
    unmeasurable_specs: Optional[list[str]] = Field(default=None, description="无法从原型判断的规格列表")


class DocumentInfo(BaseModel):
    """已上传的 PRD 文档信息"""
    document_id: str = Field(description="文档唯一标识")
    file_name: str = Field(description="文件名")
    file_path: str = Field(description="文件路径")
    uploaded_at: datetime = Field(default_factory=datetime.now, description="上传时间")
    chunk_ids: list[str] = Field(default_factory=list, description="切分后的 chunk IDs")
    metadata: dict = Field(default_factory=dict, description="文档元数据")


class ReviewSession(BaseModel):
    """审查会话 - 管理 PRD 和原型图的关联"""
    session_id: str = Field(description="会话唯一标识")
    document: Optional[DocumentInfo] = Field(default=None, description="关联的 PRD 文档")
    prototypes: list[PrototypeDescription] = Field(default_factory=list, description="关联的原型图列表")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="最后更新时间")
    status: str = Field(default="pending", description="会话状态: pending, reviewing, completed")


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