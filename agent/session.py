"""审查会话管理器 - 管理 PRD 和原型图的关联、绑定/解绑"""

import uuid
import logging
import sqlite3
import json
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from agent.schemas import (
    ReviewSession,
    DocumentInfo,
    PrototypeDescription,
)
from vec_db.embeddings.chunker import ProductStandardChunker
from vec_db.embeddings.bge_m3_embedding import BgeM3Embeddings
from vec_db.store.prototype_store import PrototypeStore


logger = logging.getLogger(__name__)


def _row_val(row: sqlite3.Row, key: str, default=None):
    """安全获取 sqlite3.Row 的值（兼容旧数据库中不存在的列）"""
    try:
        val = row[key]
        return val if val is not None else default
    except (IndexError, KeyError):
        return default


class ReviewSessionManager:
    """
    审查会话管理器

    功能：
    - 创建审查会话
    - 上传/注册 PRD 文档（带自动切分）
    - 上传/注册原型图（支持绑定 document_id）
    - 绑定/解绑 PRD 与原型图的关联
    - 支持纠错：已绑定的可以解绑重新绑定
    - SQLite 持久化 session 元数据
    """

    def __init__(self, chroma_persist_dir: str = "./chroma_data"):
        self.sessions: dict[str, ReviewSession] = {}
        self.chroma_persist_dir = chroma_persist_dir
        self.prototype_store = PrototypeStore()
        self.embedder = BgeM3Embeddings()
        self.chunker = ProductStandardChunker(chunk_size=600, overlap=90)

        # SQLite 持久化
        self._db_path = Path(chroma_persist_dir) / "sessions.db"
        self._init_db()
        self._load_sessions()

    @contextmanager
    def _get_db(self):
        """获取数据库连接上下文"""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库表（支持旧数据库迁移）"""
        with self._get_db() as conn:
            # 创建 sessions 表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    document_id TEXT,
                    document_file_name TEXT,
                    document_file_path TEXT,
                    document_chunk_ids TEXT,
                    document_metadata TEXT
                )
            """)

            # 创建 prototypes 表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prototypes (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    document_id TEXT,
                    page_name TEXT,
                    layout TEXT DEFAULT '',
                    components TEXT DEFAULT '[]',
                    interactions TEXT DEFAULT '',
                    query_text TEXT,
                    image_path TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # 迁移旧 prototypes 表：添加新列（如果不存在）
            new_columns = [
                ("page_type", "TEXT"),
                ("fidelity_level", "TEXT"),
                ("module_path", "TEXT"),
                ("layout_type", "TEXT"),
                ("min_resolution", "TEXT"),
                ("primary_color", "TEXT"),
                ("font_hierarchy", "TEXT"),
                ("component_spacing", "TEXT"),
                ("button_types", "TEXT"),
                ("form_fields_per_row", "INTEGER"),
                ("table_column_count", "INTEGER"),
                ("table_page_size", "INTEGER"),
                ("popup_types", "TEXT"),
                ("filter_default_fields", "INTEGER"),
                ("filter_total_fields", "INTEGER"),
                ("stat_card_count", "INTEGER"),
                ("component_library", "TEXT"),
                ("currency_format", "TEXT"),
                ("date_format", "TEXT"),
                ("validation_type", "TEXT"),
                ("validation_rules", "TEXT"),
                ("full_content", "TEXT"),
                ("extracted_specs", "TEXT"),
                ("unmeasurable_specs", "TEXT"),
            ]

            # 获取现有列
            existing_cols = set(row[1] for row in conn.execute("PRAGMA table_info(prototypes)"))

            # 添加缺失的列
            for col_name, col_type in new_columns:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE prototypes ADD COLUMN {col_name} {col_type}")

            conn.commit()

    def _load_sessions(self):
        """从数据库加载 session"""
        with self._get_db() as conn:
            rows = conn.execute("SELECT * FROM sessions").fetchall()
            for row in rows:
                session_id = row["session_id"]

                # 重建 DocumentInfo
                document = None
                if row["document_id"]:
                    chunk_ids = json.loads(row["document_chunk_ids"] or "[]")
                    metadata = json.loads(row["document_metadata"] or "{}")
                    document = DocumentInfo(
                        document_id=row["document_id"],
                        file_name=row["document_file_name"] or "",
                        file_path=row["document_file_path"] or "",
                        chunk_ids=chunk_ids,
                        metadata=metadata,
                    )

                # 重建 Session 对象
                import datetime
                session = ReviewSession(
                    session_id=session_id,
                    status=row["status"],
                    created_at=datetime.datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.datetime.fromisoformat(row["updated_at"]),
                    document=document,
                )

                # 加载 prototypes
                proto_rows = conn.execute(
                    "SELECT * FROM prototypes WHERE session_id = ?", (session_id,)
                ).fetchall()
                for pr in proto_rows:
                    # 安全获取所有字段（兼容旧数据库中没有新列的情况）
                    font_h_str = _row_val(pr, "font_hierarchy")
                    validation_str = _row_val(pr, "validation_rules")
                    extracted_str = _row_val(pr, "extracted_specs")
                    unmeas_str = _row_val(pr, "unmeasurable_specs")
                    button_str = _row_val(pr, "button_types")
                    popup_str = _row_val(pr, "popup_types")
                    components_str = _row_val(pr, "components")

                    session.prototypes.append(PrototypeDescription(
                        id=_row_val(pr, "id", ""),
                        name=_row_val(pr, "name", ""),
                        document_id=_row_val(pr, "document_id"),
                        page_name=_row_val(pr, "page_name", ""),
                        layout=_row_val(pr, "layout", ""),
                        components=json.loads(components_str or "[]"),
                        interactions=_row_val(pr, "interactions", ""),
                        query_text=_row_val(pr, "query_text", ""),
                        image_path=_row_val(pr, "image_path", ""),
                        # New fields (may be None for old data)
                        page_type=_row_val(pr, "page_type"),
                        fidelity_level=_row_val(pr, "fidelity_level"),
                        module_path=_row_val(pr, "module_path"),
                        layout_type=_row_val(pr, "layout_type"),
                        min_resolution=_row_val(pr, "min_resolution"),
                        primary_color=_row_val(pr, "primary_color"),
                        font_hierarchy=json.loads(font_h_str) if font_h_str else None,
                        component_spacing=_row_val(pr, "component_spacing"),
                        button_types=button_str.split(",") if button_str else None,
                        form_fields_per_row=_row_val(pr, "form_fields_per_row"),
                        table_column_count=_row_val(pr, "table_column_count"),
                        table_page_size=_row_val(pr, "table_page_size"),
                        popup_types=popup_str.split(",") if popup_str else None,
                        filter_default_fields=_row_val(pr, "filter_default_fields"),
                        filter_total_fields=_row_val(pr, "filter_total_fields"),
                        stat_card_count=_row_val(pr, "stat_card_count"),
                        component_library=_row_val(pr, "component_library"),
                        currency_format=_row_val(pr, "currency_format"),
                        date_format=_row_val(pr, "date_format"),
                        validation_type=_row_val(pr, "validation_type"),
                        validation_rules=json.loads(validation_str) if validation_str else None,
                        full_content=_row_val(pr, "full_content"),
                        extracted_specs=json.loads(extracted_str) if extracted_str else None,
                        unmeasurable_specs=unmeas_str.split(",") if unmeas_str else None,
                    ))

                self.sessions[session_id] = session

        logger.info(f"Loaded {len(self.sessions)} sessions from database")

    def _save_session(self, session: ReviewSession):
        """保存单个 session 到数据库"""
        with self._get_db() as conn:
            document_chunk_ids = json.dumps([c for c in session.document.chunk_ids]) if session.document else "[]"
            document_metadata = json.dumps(session.document.metadata if session.document else {})

            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, status, created_at, updated_at, document_id, document_file_name,
                 document_file_path, document_chunk_ids, document_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.status,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                session.document.document_id if session.document else None,
                session.document.file_name if session.document else None,
                session.document.file_path if session.document else None,
                document_chunk_ids,
                document_metadata,
            ))
            conn.commit()

    def _save_prototypes(self, session_id: str, prototypes: list[PrototypeDescription]):
        """保存 prototypes 到数据库"""
        with self._get_db() as conn:
            # 删除旧的 prototypes
            conn.execute("DELETE FROM prototypes WHERE session_id = ?", (session_id,))

            # 插入新的
            for p in prototypes:
                conn.execute(f"""
                    INSERT INTO prototypes
                    (id, session_id, name, document_id, page_name, layout, components,
                     interactions, query_text, image_path,
                     page_type, fidelity_level, module_path,
                     layout_type, min_resolution,
                     primary_color, font_hierarchy, component_spacing,
                     button_types, form_fields_per_row, table_column_count, table_page_size,
                     popup_types, filter_default_fields, filter_total_fields, stat_card_count,
                     component_library,
                     currency_format, date_format,
                     validation_type, validation_rules,
                     full_content, extracted_specs, unmeasurable_specs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    p.id,
                    session_id,
                    p.name,
                    p.document_id,
                    p.page_name,
                    p.layout,
                    json.dumps(p.components),
                    p.interactions,
                    p.query_text,
                    p.image_path,
                    # New fields
                    p.page_type,
                    p.fidelity_level,
                    p.module_path,
                    p.layout_type,
                    p.min_resolution,
                    p.primary_color,
                    json.dumps(p.font_hierarchy) if p.font_hierarchy else None,
                    p.component_spacing,
                    ",".join(p.button_types) if p.button_types else None,
                    p.form_fields_per_row,
                    p.table_column_count,
                    p.table_page_size,
                    ",".join(p.popup_types) if p.popup_types else None,
                    p.filter_default_fields,
                    p.filter_total_fields,
                    p.stat_card_count,
                    p.component_library,
                    p.currency_format,
                    p.date_format,
                    p.validation_type,
                    json.dumps(p.validation_rules) if p.validation_rules else None,
                    p.full_content,
                    json.dumps(p.extracted_specs) if p.extracted_specs else None,
                    ",".join(p.unmeasurable_specs) if p.unmeasurable_specs else None,
                ))
            conn.commit()

    def create_session(self) -> str:
        """创建新的审查会话"""
        session_id = str(uuid.uuid4())[:8]
        session = ReviewSession(session_id=session_id)
        self.sessions[session_id] = session
        self._save_session(session)
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
            # 批量向量化，避免循环内逐条插入 ChromaDB
            contents = [c.content for c in chunk_results]
            embeddings = self.embedder.embed_documents(contents)

            # 预生成所有 chunk_id 和 metadata
            ids_to_store = []
            docs_to_store = []
            emb_to_store = []
            meta_to_store = []

            import hashlib
            for i, chunk_result in enumerate(chunk_results):
                hash_input = f"{chunk_result.content}|{chunk_result.metadata.get('header_path', '')}|{i}"
                chunk_id = hashlib.sha256(hash_input.encode()).hexdigest()[:32]
                chunk_metadata = {
                    "document_id": document_id,
                    "header_path": chunk_result.metadata.get("header_path", ""),
                    "heading": chunk_result.metadata.get("heading", ""),
                    "chunk_index": i,
                    **chunk_result.metadata,
                }
                ids_to_store.append(chunk_id)
                docs_to_store.append(chunk_result.content)
                emb_to_store.append(embeddings[i])
                meta_to_store.append(chunk_metadata)
                stored_chunk_ids.append(chunk_id)

            # 一次批量插入
            chroma_store.store(
                ids=ids_to_store,
                documents=docs_to_store,
                embeddings=emb_to_store,
                metadatas=meta_to_store,
            )

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
        self._save_session(session)

        logger.info(f"Registered PRD {document_id} to session {session_id}: {len(stored_chunk_ids)} chunks")
        return doc_info

    def register_prototype(
        self,
        session_id: str,
        image_path: str,
        name: str,
        document_id: Optional[str] = None,
    ) -> PrototypeDescription:
        """
        注册原型图到会话（仅绑定，不分析）

        注意：原型图的分析和丰富存储在 review 阶段通过 enrich_prototype 工具完成。

        Args:
            session_id: 会话 ID
            image_path: 原型图文件路径
            name: 原型图名称（如"登录页"）
            document_id: 可选，关联的 PRD document_id

        Returns:
            PrototypeDescription: 最小化的原型图描述
        """
        session = self._get_session(session_id)

        # 生成 id
        prototype_id = str(uuid.uuid4())[:16]

        # 创建最小化描述（不分析，layout/components 等在 review 阶段填充）
        description = PrototypeDescription(
            id=prototype_id,
            name=name,
            document_id=document_id,
            page_name=name,
            layout="",
            components=[],
            interactions="",
            query_text=name,
            image_path=image_path,
        )

        # 添加到会话（不存储到向量库，由 review 阶段的 enrich_prototype 处理）
        session.prototypes.append(description)
        session.updated_at = __import__("datetime").datetime.now()
        self._save_session(session)
        self._save_prototypes(session_id, session.prototypes)

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
                self._save_session(session)
                self._save_prototypes(session_id, session.prototypes)
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
                self._save_session(session)
                self._save_prototypes(session_id, session.prototypes)
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
