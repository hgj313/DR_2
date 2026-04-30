"""原型图描述存储 - 使用 ChromaDB"""

import uuid
from typing import Optional

from vec_db.store.chroma_store import ChromaStore
from agent.schemas import PrototypeDescription


class PrototypeStore:
    """原型图描述存储管理"""

    def __init__(
        self,
        collection_name: str = "prototype_descriptions",
        persist_directory: str = "./chroma_data",
        use_remote: bool = False,
    ):
        self.chroma_store = ChromaStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            use_remote=use_remote,
        )

    def store(
        self,
        description: PrototypeDescription,
        embedding: list[float],
    ) -> str:
        """存储原型描述（防重复：基于 document_id + page_name）"""
        from langchain_core.documents import Document

        # 检查是否已存在（基于 document_id + page_name）
        existing = self.chroma_store._collection.get(
            where={
                "$and": [
                    {"document_id": description.document_id},
                    {"page_name": description.page_name}
                ]
            }
        )

        if existing and existing.get("ids"):
            # 已存在，更新而非重复插入
            doc_id = existing["ids"][0]
            self.chroma_store._collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[description.query_text],
                metadatas=[{
                    "id": doc_id,
                    "name": description.name,
                    "document_id": description.document_id,
                    "page_name": description.page_name,
                    "layout": description.layout,
                    "components": ",".join(description.components),
                    "interactions": description.interactions,
                    "query_text": description.query_text,
                    "image_path": description.image_path,
                }]
            )
            return doc_id

        # 新增
        doc_id = description.id or str(uuid.uuid4())

        doc = Document(
            page_content=description.query_text,
            metadata={
                "id": doc_id,
                "name": description.name,
                "document_id": description.document_id,
                "page_name": description.page_name,
                "layout": description.layout,
                "components": ",".join(description.components),
                "interactions": description.interactions,
                "query_text": description.query_text,
                "image_path": description.image_path,
            }
        )

        # 使用 upsert 存储
        self.chroma_store._collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc.page_content],
            metadatas=[doc.metadata],
        )

        return doc_id

    def update_document_id(self, prototype_id: str, new_document_id: Optional[str]) -> bool:
        """更新原型图的 document_id（用于绑定/解绑）"""
        try:
            # 查找该 prototype
            existing = self.chroma_store._collection.get(
                where={"id": prototype_id}
            )

            if not existing or not existing.get("ids"):
                return False

            # 获取现有 metadata
            old_meta = existing["metadatas"][0] if existing.get("metadatas") else {}
            old_meta["document_id"] = new_document_id

            # 更新
            self.chroma_store._collection.update(
                ids=[prototype_id],
                metadatas=[old_meta],
            )
            return True
        except Exception as e:
            print(f"Error updating prototype document_id: {e}")
            return False

    def retrieve(
        self,
        query_embedding: list[float],
        k: int = 5,
        document_id: Optional[str] = None,
    ) -> list[PrototypeDescription]:
        """检索原型描述（通过 name 或 query_text 语义检索）"""
        results = self.chroma_store.similarity_search(
            embedding=query_embedding,
            k=k,
        )

        descriptions = []
        for doc, score in results:
            meta = doc.metadata
            if document_id and meta.get("document_id") != document_id:
                continue

            descriptions.append(PrototypeDescription(
                id=meta.get("id", ""),
                name=meta.get("name", ""),
                document_id=meta.get("document_id", ""),
                page_name=meta.get("page_name", ""),
                layout=meta.get("layout", ""),
                components=meta.get("components", "").split(",") if meta.get("components") else [],
                interactions=meta.get("interactions", ""),
                query_text=meta.get("query_text", ""),
                image_path=meta.get("image_path", ""),
            ))

        return descriptions