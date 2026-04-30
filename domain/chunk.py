from datetime import datetime
from dataclasses import dataclass, field

from domain.value_objects import ChunkId, DocumentId


@dataclass
class ChunkMetadata:
    """
    标准chunk元数据 - 存入向量数据库的完整元数据结构
    
    包含：
    - 身份标识：chunk_id, document_id, content_hash
    - 层级结构：header_path, heading, heading_level
    - 位置信息：chunk_index, chunk_start_index, section_start_index, total_chunks
    - 内容类型：chunk_type
    - 来源信息：source, created_at
    """
    chunk_id: str = ""
    document_id: str = ""
    content_hash: str = ""  # 内容哈希，用于去重
    
    header_path: str = ""
    heading: str = ""
    heading_level: int = 0
    
    chunk_index: int = 0
    chunk_start_index: int = 0
    section_start_index: int = 0
    total_chunks: int = 0
    
    chunk_type: str = "text"
    source: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    custom_metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """转换为字典，用于存储到向量数据库"""
        result = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content_hash": self.content_hash,
            "header_path": self.header_path,
            "heading": self.heading,
            "heading_level": self.heading_level,
            "chunk_index": self.chunk_index,
            "chunk_start_index": self.chunk_start_index,
            "section_start_index": self.section_start_index,
            "total_chunks": self.total_chunks,
            "chunk_type": self.chunk_type,
            "source": self.source,
            "created_at": self.created_at,
        }
        if self.custom_metadata:
            result.update(self.custom_metadata)
        return result


@dataclass
class Chunk:
    """
    文档分块
    """
    id: ChunkId
    document_id: DocumentId
    content: str
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    embedding: list[list[float]] | None = None
