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