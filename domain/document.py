from datetime import datetime
from dataclasses import dataclass, field

from langchain_core.documents import Document

from domain.value_objects import DocumentId


@dataclass
class DesignDocument(Document):
    """
    设计文档类，继承自 langchain Document
    
    新增字段：
    - id: 文档唯一标识
    - created_at: 创建时间
    """

    id: DocumentId = None
    created_at: datetime = field(default_factory=datetime.now)
