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
    layout: str = Field(description="布局描述")
    components: list[str] = Field(description="组件列表")
    interactions: str = Field(description="交互描述")
    query_text: str = Field(description="用于检索的文本（聚合 name + page_name + layout）")
    image_path: str = Field(description="原型图文件路径")


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