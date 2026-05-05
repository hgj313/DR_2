"""审查会话管理器 - 管理 PRD 和原型图的关联、绑定/解绑"""

import uuid
import logging
from pathlib import Path
from typing import Optional

from agent.schemas import (
    ReviewSession,
    DocumentInfo,
    PrototypeDescription,
)
from vec_db.embeddings.chunker import ProductStandardChunker
from vec_db.embeddings.bge_m3_embedding import BgeM3Embeddings
from vec_db.store.prototype_store import PrototypeStore


logger = logging.getLogger(__name__)


class ReviewSessionManager:
    """
    审查会话管理器

    功能：
    - 创建审查会话
    - 上传/注册 PRD 文档（带自动切分）
    - 上传/注册原型图（支持绑定 document_id）
    - 绑定/解绑 PRD 与原型图的关联
    - 支持纠错：已绑定的可以解绑重新绑定
    """

    def __init__(self, chroma_persist_dir: str = "./chroma_data"):
        self.sessions: dict[str, ReviewSession] = {}
        self.chroma_persist_dir = chroma_persist_dir
        self.prototype_store = PrototypeStore()
        self.embedder = BgeM3Embeddings()
        self.chunker = ProductStandardChunker(chunk_size=600, overlap=90)

    def create_session(self) -> str:
        """创建新的审查会话"""
        session_id = str(uuid.uuid4())[:8]
        session = ReviewSession(session_id=session_id)
        self.sessions[session_id] = session
        logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[ReviewSession]:
        """获取会话"""
        return self.sessions.get(session_id)

    def register_prd(
        self,
        session_id: str,
        file_path: str,
        metadata: Optional[dict] = None,
    ) -> DocumentInfo:
        """
        注册 PRD 文档到会话（自动切分 + 存储）

        Args:
            session_id: 会话 ID
            file_path: PRD 文件路径（支持 .md）
            metadata: 额外元数据

        Returns:
            DocumentInfo: 文档信息
        """
        session = self._get_session(session_id)

        # 读取文件
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # 生成 document_id
        document_id = str(uuid.uuid4())[:16]

        # 切分文档
        chunk_results = self.chunker.chunk(content, metadata or {})

        # 存储 chunks 到 ChromaDB（使用 document_store 的逻辑）
        from vec_db.store.chroma_store import ChromaStore
        chroma_store = ChromaStore(
            collection_name="prd_documents",
            persist_directory=self.chroma_persist_dir,
            use_remote=False,
        )

        stored_chunk_ids = []
        if chunk_results:
            contents = [c.content for c in chunk_results]
            embeddings = self.embedder.embed_documents(contents)

            for i, chunk_result in enumerate(chunk_results):
                import hashlib
                hash_input = f"{chunk_result.content}|{chunk_result.metadata.get('header_path', '')}|{i}"
                chunk_id = hashlib.sha256(hash_input.encode()).hexdigest()[:32]

                chunk_metadata = {
                    "document_id": document_id,
                    "header_path": chunk_result.metadata.get("header_path", ""),
                    "heading": chunk_result.metadata.get("heading", ""),
                    "chunk_index": i,
                    **chunk_result.metadata,
                }
                #循环里面插入数据库性能极差，后续第二次改进-2026.5.5（hgj)
                chroma_store.store(
                    ids=[chunk_id],
                    documents=[chunk_result.content],
                    embeddings=[embeddings[i]],
                    metadatas=[chunk_metadata],
                )
                stored_chunk_ids.append(chunk_id)

        # 创建 DocumentInfo
        doc_info = DocumentInfo(
            document_id=document_id,
            file_name=path.name,
            file_path=str(path.absolute()),
            chunk_ids=stored_chunk_ids,
            metadata=metadata or {},
        )

        # 更新会话
        session.document = doc_info
        session.updated_at = __import__("datetime").datetime.now()

        logger.info(f"Registered PRD {document_id} to session {session_id}: {len(stored_chunk_ids)} chunks")
        return doc_info

    def register_prototype(
        self,
        session_id: str,
        image_path: str,
        name: str,
        document_id: Optional[str] = None,
        layout: str = "",
        components: list[str] = None,
        interactions: str = "",
    ) -> PrototypeDescription:
        """
        注册原型图到会话

        Args:
            session_id: 会话 ID
            image_path: 原型图文件路径
            name: 原型图名称（如"登录页"）
            document_id: 可选，关联的 PRD document_id
            layout: 布局描述
            components: 组件列表
            interactions: 交互描述

        Returns:
            PrototypeDescription: 原型图描述
        """
        session = self._get_session(session_id)
        if components is None:
            components = []

        # 生成 id
        prototype_id = str(uuid.uuid4())[:16]

        # 生成 query_text
        query_text = f"{name} {layout} {' '.join(components)}"

        # 生成 embedding
        embedding = self.embedder.embed_query(query_text)

        # 创建描述对象
        description = PrototypeDescription(
            id=prototype_id,
            name=name,
            document_id=document_id,
            page_name=name,
            layout=layout,
            components=components,
            interactions=interactions,
            query_text=query_text,
            image_path=image_path,
        )

        # 存储到 PrototypeStore
        self.prototype_store.store(description, embedding)

        # 添加到会话
        session.prototypes.append(description)
        session.updated_at = __import__("datetime").datetime.now()

        logger.info(f"Registered prototype {prototype_id} to session {session_id}, bound to doc {document_id}")
        return description

    def bind_prototype_to_document(
        self,
        session_id: str,
        prototype_id: str,
        document_id: str,
    ) -> bool:
        """
        将原型图绑定到 PRD 文档

        Args:
            session_id: 会话 ID
            prototype_id: 原型图 ID
            document_id: PRD 文档 ID

        Returns:
            bool: 是否成功
        """
        session = self._get_session(session_id)

        # 验证 document_id 存在
        if session.document and session.document.document_id != document_id:
            logger.warning(f"Document {document_id} not in session {session_id}")
            return False

        # 找到原型图并更新
        for prototype in session.prototypes:
            if prototype.id == prototype_id:
                prototype.document_id = document_id

                # 更新 ChromaDB 中的记录
                self.prototype_store.update_document_id(prototype_id, document_id)

                session.updated_at = __import__("datetime").datetime.now()
                logger.info(f"Bound prototype {prototype_id} to document {document_id}")
                return True

        logger.warning(f"Prototype {prototype_id} not found in session {session_id}")
        return False

    def unbind_prototype(self, session_id: str, prototype_id: str) -> bool:
        """
        解除原型图与 PRD 文档的绑定

        Args:
            session_id: 会话 ID
            prototype_id: 原型图 ID

        Returns:
            bool: 是否成功
        """
        session = self._get_session(session_id)

        for prototype in session.prototypes:
            if prototype.id == prototype_id:
                old_doc_id = prototype.document_id
                prototype.document_id = None

                # 更新 ChromaDB 中的记录
                self.prototype_store.update_document_id(prototype_id, None)

                session.updated_at = __import__("datetime").datetime.now()
                logger.info(f"Unbound prototype {prototype_id} from document {old_doc_id}")
                return True

        logger.warning(f"Prototype {prototype_id} not found in session {session_id}")
        return False

    def upload_session(
        self,
        session_id: str,
        prd_file: str = None,
        prototype_files: list = None,
        metadata: dict = None,
    ) -> dict:
        """
        同时上传 PRD 和原型图，自动建立关联

        Args:
            session_id: 会话 ID
            prd_file: PRD 文件路径（支持 .md）
            prototype_files: 原型图文件列表，格式：
                - [{"path": "路径", "name": "名称"}, ...]
                - 或直接传 ["/path/to/img.png", ...]（自动用文件名作为 name）
            metadata: 文档元数据

        Returns:
            dict: {
                "document": DocumentInfo 或 None,
                "prototypes": [PrototypeDescription, ...],
                "bindings": [(prototype_id, document_id), ...]  # 自动绑定的记录
            }
        """
        session = self._get_session(session_id)
        if metadata is None:
            metadata = {}
        if prototype_files is None:
            prototype_files = []

        result = {
            "document": None,
            "prototypes": [],
            "bindings": [],
        }

        # 1. 注册 PRD（如果提供）
        document_id = None
        if prd_file:
            try:
                doc_info = self.register_prd(session_id, prd_file, metadata)
                result["document"] = doc_info
                document_id = doc_info.document_id
            except Exception as e:
                logger.error(f"Failed to register PRD: {e}")

        # 2. 注册原型图并自动绑定
        for item in prototype_files:
            # 支持两种格式：{"path": x, "name": y} 或 直接是路径字符串
            if isinstance(item, dict):
                image_path = item.get("path", "")
                name = item.get("name", Path(image_path).stem)
            else:
                image_path = item
                name = Path(image_path).stem

            try:
                proto = self.register_prototype(
                    session_id=session_id,
                    image_path=image_path,
                    name=name,
                    document_id=document_id,  # 自动绑定（可能为 None）
                )
                result["prototypes"].append(proto)

                # 如果有 document_id，记录绑定
                if document_id:
                    result["bindings"].append((proto.id, document_id))

            except Exception as e:
                logger.error(f"Failed to register prototype {image_path}: {e}")

        return result

    def get_session_summary(self, session_id: str) -> str:
        """获取会话摘要"""
        session = self._get_session(session_id)
        if not session:
            return f"Session {session_id} not found"

        lines = [
            f"Session: {session.session_id}",
            f"Status: {session.status}",
            f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        if session.document:
            lines.append(f"PRD Document: {session.document.file_name}")
            lines.append(f"  - Document ID: {session.document.document_id}")
            lines.append(f"  - Chunks: {len(session.document.chunk_ids)}")
        else:
            lines.append("PRD Document: (not uploaded)")

        lines.append("")
        lines.append(f"Prototypes: {len(session.prototypes)}")
        for p in session.prototypes:
            bind_status = f"bound to {p.document_id}" if p.document_id else "(unbound)"
            lines.append(f"  - {p.name} [{p.id}] {bind_status}")

        return "\n".join(lines)

    def _get_session(self, session_id: str) -> ReviewSession:
        """获取会话，不存在则抛异常"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session


# 全局单例
_session_manager: Optional[ReviewSessionManager] = None


def get_session_manager() -> ReviewSessionManager:
    """获取全局会话管理器"""
    global _session_manager
    if _session_manager is None:
        _session_manager = ReviewSessionManager()
    return _session_manager
