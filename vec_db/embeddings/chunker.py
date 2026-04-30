"""
文档切分器模块

提供多种文档切分策略，支持：
- 产品设计标准文档（Markdown）- 技术教程文档（HTML）
- 通用文本

参考设计：
- D-01: Chunk size 500-800 tokens (默认 600)
- D-02: Overlap 15-20% (默认 90 tokens)
- D-03: 按标题层级 + 段落切分
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from domain.document import DesignDocument


@dataclass
class ChunkResult:
    """切分结果"""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    chunk_type: str = "text"
    header_path: list[str] = field(default_factory=list)


class BaseChunker(ABC):
    """切分器基类"""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[ChunkResult]:
        """
        执行切分

        Args:
            text: 待切分文本
            metadata: 基础元数据

        Returns:
            切分结果列表
        """
        pass

    def _create_metadata(
        self,
        base_metadata: dict[str, Any] | None,
        **kwargs
    ) -> dict[str, Any]:
        """创建元数据"""
        meta = dict(base_metadata) if base_metadata else {}
        meta.update(kwargs)
        return meta


class ProductStandardChunker(BaseChunker):
    """
    产品设计标准文档专用切分器

    特点：
    - 基于 token 计数切分（更准确）
    - 基于 Markdown 标题层级切分
    - 保留 header_path 层级结构
    - 表格作为独立 chunk 保留
    - 支持重叠区域保持上下文连贯性

    参考 D-01, D-02, D-03 规范
    """

    DEFAULT_SEPARATORS = [
        "\n## ",    # 二级标题
        "\n# ",     # 一级标题
        "\n\n",     # 段落
        "\n",        # 换行
        "。",        # 句号
        "！",        # 感叹号
        "？",        # 问号
        "；",        # 分号
        "，",        # 逗号
        " ",         # 空格
    ]

    def __init__(
        self,
        chunk_size: int = 600,
        overlap: int = 90,
        min_section_level: int = 1,
        separate_tables: bool = True,
        separators: Optional[list[str]] = None,
        length_function: Optional[Callable[[str], int]] = None,
    ):
        """
        初始化切分器

        Args:
            chunk_size: 目标 chunk 大小（tokens，默认 600 per D-01）
            overlap: chunk 之间的重叠大小（tokens，默认 90 per D-02）
            min_section_level: 最小保留的标题级别（1 = #）
            separate_tables: 是否将表格单独作为 chunk
            separators: 自定义分隔符列表
            length_function: token 计数函数
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_section_level = min_section_level
        self.separate_tables = separate_tables
        self.separators = separators or self.DEFAULT_SEPARATORS

        if length_function is None:
            length_function = self._token_count

        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=length_function,
            add_start_index=True,
        )

    @staticmethod
    def _token_count(text: str) -> int:
        """
        估算 token 数量

        中文约 1-2 字符/token，英文约 4 字符/token
        这里使用简化估算：中文按 1.5，英文按 4

        Args:
            text: 输入文本

        Returns:
            估算的 token 数量
        """
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.25)

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None
    ) -> list[ChunkResult]:
        """
        执行切分

        Args:
            text: Markdown 格式的文档内容
            metadata: 基础元数据

        Returns:
            切分结果列表
        """
        if not text or not text.strip():
            return []

        base_meta = metadata or {}

        sections, section_starts, header_paths = self._split_on_headers(text)

        all_chunks = []
        chunk_index = 0

        for idx, section_text in enumerate(sections):
            if not section_text.strip():
                continue

            path = header_paths[idx]
            section_start = section_starts[idx]

            section_meta = {
                "header_path": "/".join(path) if path else "",
                "heading": path[-1] if path else "",
                "heading_level": len(path) if path else 0,
            }

            combined_meta = self._create_metadata(base_meta, **section_meta)

            section_tokens = self._token_count(section_text)

            if section_tokens <= self.chunk_size:
                chunk = ChunkResult(
                    content=section_text.strip(),
                    metadata=combined_meta,
                    chunk_index=chunk_index,
                    chunk_type="text",
                    header_path=path
                )
                chunk.metadata["chunk_start_index"] = section_start
                all_chunks.append(chunk)
                chunk_index += 1
            else:
                section_docs = self.splitter.create_documents(
                    [section_text],
                    metadatas=[combined_meta]
                )

                for section_doc in section_docs:
                    chunk = ChunkResult(
                        content=section_doc.page_content,
                        metadata={
                            **section_doc.metadata,
                            **section_meta,
                            "chunk_start_index": section_start + section_doc.metadata.get("start_index", 0)
                        },
                        chunk_index=chunk_index,
                        chunk_type="text",
                        header_path=path
                    )
                    all_chunks.append(chunk)
                    chunk_index += 1

        if self.separate_tables:
            table_chunks = self._extract_and_create_table_chunks(text, base_meta)
            for i, table_chunk in enumerate(table_chunks):
                table_chunk.chunk_index = chunk_index + i
            all_chunks.extend(table_chunks)

        all_chunks = self._merge_small_chunks(all_chunks)

        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.metadata["total_chunks"] = len(all_chunks)

        return all_chunks

    def _merge_small_chunks(self, chunks: list[ChunkResult]) -> list[ChunkResult]:
        """合并只有标题的小 chunks"""
        if not chunks:
            return chunks

        merged = []
        current_chunks = []

        for chunk in chunks:
            if chunk.chunk_type == "table":
                if current_chunks:
                    merged_chunk = self._combine_chunks(current_chunks)
                    if merged_chunk:
                        merged.append(merged_chunk)
                    current_chunks = []
                merged.append(chunk)
                continue

            content_tokens = self._token_count(chunk.content)
            is_small_header_only = content_tokens < 20 and chunk.content.startswith("#")

            if is_small_header_only:
                current_chunks.append(chunk)
            else:
                if current_chunks:
                    merged_chunk = self._combine_chunks(current_chunks)
                    if merged_chunk:
                        merged.append(merged_chunk)
                    current_chunks = []
                current_chunks.append(chunk)

        if current_chunks:
            merged_chunk = self._combine_chunks(current_chunks)
            if merged_chunk:
                merged.append(merged_chunk)

        return merged

    def _combine_chunks(self, chunks: list[ChunkResult]) -> ChunkResult | None:
        """将多个 chunks 合并为一个"""
        if not chunks:
            return None

        combined_content = "\n\n".join(c.content for c in chunks)
        if not combined_content.strip():
            return None

        first_chunk = chunks[0]
        combined_metadata = dict(first_chunk.metadata)

        return ChunkResult(
            content=combined_content,
            metadata=combined_metadata,
            chunk_type="text",
            header_path=first_chunk.header_path
        )

    def _split_on_headers(self, text: str) -> tuple[list[str], list[int], list[list[str]]]:
        """
        按标题切分文本，同时保留标题层级结构

        关键设计：
        - 每个 section 从一个标题开始，到下一个同级或更高级标题结束
        - section 内容包含：标题 + 后续内容
        - header_path 反映当前 section 的标题路径

        Args:
            text: 文本内容

        Returns:
            Tuple of (sections, start_positions, header_paths) where:
            - sections: 每个 section 的文本内容（包含标题）
            - start_positions: 每个 section 在原文本中的字符偏移
            - header_paths: 标题路径列表，如 [["文档标题"], ["文档标题", "第一章"]] 
        """
        sections = []
        start_positions = []
        header_paths = []

        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

        header_stack: list[tuple[int, str]] = []
        current_section_start = 0
        current_section_lines = []
        in_section = False

        for match in header_pattern.finditer(text):
            level = len(match.group(1))
            header_content = match.group(2).strip()
            header_start = match.start()

            if in_section:
                prev_header = header_stack[-1] if header_stack else None
                if prev_header and level <= prev_header[0]:
                    section_text = "\n".join(current_section_lines)
                    if section_text.strip():
                        sections.append(section_text)
                        start_positions.append(current_section_start)
                        header_paths.append([h[1] for h in header_stack])
                    current_section_lines = []

            header_stack = [h for h in header_stack if h[0] < level]
            header_stack.append((level, header_content))

            if not in_section:
                in_section = True

            current_section_start = header_start
            current_section_lines.append(match.group(0))

            next_match = match
            while True:
                next_pos = next_match.end()
                next_match = header_pattern.search(text, next_pos)
                if next_match:
                    section_content = text[current_section_start + len(match.group(0)):next_match.start()]
                    current_section_lines.append(section_content)
                    break
                else:
                    section_content = text[current_section_start + len(match.group(0)):]
                    current_section_lines.append(section_content)
                    break

        if current_section_lines:
            section_text = "\n".join(current_section_lines)
            if section_text.strip():
                sections.append(section_text)
                start_positions.append(current_section_start)
                header_paths.append([h[1] for h in header_stack])

        return sections, start_positions, header_paths

    def _extract_and_create_table_chunks(
        self,
        text: str,
        base_metadata: dict[str, Any]
    ) -> list[ChunkResult]:
        """提取表格并创建独立的 table chunks"""
        table_chunks = []
        tables = self._extract_tables(text)

        for i, table in enumerate(tables):
            table_chunks.append(ChunkResult(
                content=table,
                metadata=self._create_metadata(
                    base_metadata,
                    header_path="",
                    heading=f"表格 {i+1}",
                    heading_level=0,
                    chunk_type="table",
                    table_index=i
                ),
                chunk_type="table",
                header_path=[]
            ))

        return table_chunks

    def _extract_tables(self, text: str) -> list[str]:
        """提取 Markdown 表格"""
        tables = []
        lines = text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]
            if re.match(r'^\|.+\|$', line.strip()):
                table_lines = []
                while i < len(lines) and re.match(r'^\|.+\|$', lines[i].strip()):
                    table_lines.append(lines[i])
                    i += 1
                tables.append('\n'.join(table_lines))
                continue
            i += 1

        return tables


class RecursiveChunker(BaseChunker):
    """
    递归字符切分器（基于 token 计数）

    特点：
    - 按层级分隔符递归切分
    - 使用 token 计数（更准确）
    - 适合各种文本类型
    - 保持语义完整性
    """

    DEFAULT_SEPARATORS = [
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "；",
        "，",
        " ",
        "",
    ]

    def __init__(
        self,
        chunk_size: int = 600,
        overlap: int = 90,
        separators: Optional[list[str]] = None,
        length_function: Optional[Callable[[str], int]] = None,
    ):
        """
        初始化切分器

        Args:
            chunk_size: 最大 chunk 大小（tokens）
            overlap: 重叠大小（tokens）
            separators: 分隔符列表（按优先级排序）
            length_function: token 计数函数
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

        if length_function is None:
            length_function = self._token_count

        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=length_function,
            add_start_index=True,
        )

    @staticmethod
    def _token_count(text: str) -> int:
        """估算 token 数量"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.25)

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None
    ) -> list[ChunkResult]:
        """执行递归切分"""
        if not text or not text.strip():
            return []

        docs = self.splitter.create_documents(
            [text],
            metadatas=[metadata or {}]
        )

        return [
            ChunkResult(
                content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "chunk_type": "text"
                },
                chunk_index=i,
                chunk_type="text"
            )
            for i, doc in enumerate(docs)
        ]


def create_chunker(
    doc_type: str = "product_standard",
    chunk_size: int = 600,
    overlap: int = 90,
    **kwargs
) -> BaseChunker:
    """
    创建切分器

    Args:
        doc_type: 文档类型
            - "product_standard": 产品设计标准文档
            - "tutorial": 技术教程文档
            - "general": 通用文本
        chunk_size: chunk 大小（tokens）
        overlap: 重叠大小（tokens）
        **kwargs: 其他参数传递给切分器

    Returns:
        切分器实例
    """
    chunkers = {
        "product_standard": ProductStandardChunker,
        "tutorial": RecursiveChunker,
        "general": RecursiveChunker,
    }

    chunker_class = chunkers.get(doc_type, RecursiveChunker)
    return chunker_class(
        chunk_size=chunk_size,
        overlap=overlap,
        **kwargs
    )


def chunk_documents(
    documents: list[DesignDocument],
    chunk_size: int = 600,
    overlap: int = 90,
) -> list[DesignDocument]:
    """
    便捷函数：对文档进行切分

    Args:
        documents: Document 对象列表
        chunk_size: 目标 chunk 大小（tokens，默认 600）
        overlap: 重叠大小（tokens，默认 90）

    Returns:
        切分后的 Document 对象列表
    """
    chunker = ProductStandardChunker(chunk_size=chunk_size, overlap=overlap)

    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk(doc.page_content, doc.metadata)
        for chunk in chunks:
            all_chunks.append(DesignDocument(
                id=doc.id,
                page_content=chunk.content,
                metadata=chunk.metadata
            ))

    return all_chunks


def chunk_result_to_chunk(
    chunk_result: ChunkResult,
    chunk_id: str,
    document_id: str
) -> "Chunk":
    """
    将ChunkResult转换为Chunk（统一元数据结构）
    
    支持两种ID生成策略：
    1. UUID方案：每次生成新UUID，无法去重
    2. 内容哈希方案：基于内容生成哈希，自动去重（推荐）
    
    Args:
        chunk_result: 切分器产生的ChunkResult对象
        chunk_id: chunk唯一标识（可以是UUID或内容哈希）
        document_id: 所属文档ID
        use_content_hash: 是否在ID中使用内容哈希（True=可去重，False=UUID模式）
    
    Returns:
        Chunk: 带有标准ChunkMetadata的Chunk对象
    """
    import hashlib
    from domain.chunk import Chunk, ChunkMetadata
    
    base_metadata_keys = {
        "header_path", "heading", "heading_level",
        "chunk_start_index", "total_chunks", "chunk_type", "source"
    }
    
    custom_metadata = {
        k: v for k, v in chunk_result.metadata.items()
        if k not in base_metadata_keys
    }
    
    section_start_index = custom_metadata.pop("start_index", 0)
    
    # 生成内容哈希（用于去重）
    hash_input = f"{chunk_result.content}|{chunk_result.metadata.get('header_path', '')}|{chunk_result.chunk_index}"
    content_hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()[:32]
    
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        content=chunk_result.content,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            content_hash=content_hash,
            header_path=chunk_result.metadata.get("header_path", ""),
            heading=chunk_result.metadata.get("heading", ""),
            heading_level=chunk_result.metadata.get("heading_level", 0),
            chunk_index=chunk_result.chunk_index,
            chunk_start_index=chunk_result.metadata.get("chunk_start_index", 0),
            section_start_index=section_start_index,
            total_chunks=chunk_result.metadata.get("total_chunks", 0),
            chunk_type=chunk_result.chunk_type,
            source=chunk_result.metadata.get("source", ""),
            custom_metadata=custom_metadata
        )
    )


__all__ = [
    "ChunkResult",
    "BaseChunker",
    "ProductStandardChunker",
    "RecursiveChunker",
    "create_chunker",
    "chunk_documents",
    "chunk_result_to_chunk"
]
