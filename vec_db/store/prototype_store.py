"""原型图描述存储 - 使用 ChromaDB"""

import json
import uuid
from typing import Optional

from vec_db.store.chroma_store import ChromaStore
from agent.schemas import PrototypeDescription


def _build_metadata(description: PrototypeDescription) -> dict:
    """构建 metadata 字典，包含所有字段"""
    return {
        # Basic fields
        "id": description.id,
        "name": description.name,
        "document_id": description.document_id,
        "page_name": description.page_name,
        "layout": description.layout,
        "components": ",".join(description.components) if description.components else "",
        "interactions": description.interactions,
        "query_text": description.query_text,
        "image_path": description.image_path,
        # Page metadata
        "page_type": description.page_type,
        "fidelity_level": description.fidelity_level,
        "module_path": description.module_path,
        # Layout details
        "layout_type": description.layout_type,
        "min_resolution": description.min_resolution,
        # Visual elements
        "primary_color": description.primary_color,
        "font_hierarchy": json.dumps(description.font_hierarchy) if description.font_hierarchy else None,
        "component_spacing": description.component_spacing,
        # Component details
        "button_types": ",".join(description.button_types) if description.button_types else None,
        "form_fields_per_row": description.form_fields_per_row,
        "table_column_count": description.table_column_count,
        "table_page_size": description.table_page_size,
        "popup_types": ",".join(description.popup_types) if description.popup_types else None,
        "filter_default_fields": description.filter_default_fields,
        "filter_total_fields": description.filter_total_fields,
        "stat_card_count": description.stat_card_count,
        "component_library": description.component_library,
        # Data format
        "currency_format": description.currency_format,
        "date_format": description.date_format,
        # Validation
        "validation_type": description.validation_type,
        "validation_rules": json.dumps(description.validation_rules) if description.validation_rules else None,
        # Analysis results
        "full_content": description.full_content,
        "extracted_specs": json.dumps(description.extracted_specs) if description.extracted_specs else None,
        "unmeasurable_specs": ",".join(description.unmeasurable_specs) if description.unmeasurable_specs else None,
    }


def _parse_metadata(meta: dict) -> dict:
    """从 metadata 字典解析出 PrototypeDescription 构造参数"""
    return {
        "id": meta.get("id", ""),
        "name": meta.get("name", ""),
        "document_id": meta.get("document_id") or None,
        "page_name": meta.get("page_name", ""),
        "layout": meta.get("layout", ""),
        "components": meta.get("components", "").split(",") if meta.get("components") else [],
        "interactions": meta.get("interactions", ""),
        "query_text": meta.get("query_text", ""),
        "image_path": meta.get("image_path", ""),
        # Page metadata
        "page_type": meta.get("page_type"),
        "fidelity_level": meta.get("fidelity_level"),
        "module_path": meta.get("module_path"),
        # Layout details
        "layout_type": meta.get("layout_type"),
        "min_resolution": meta.get("min_resolution"),
        # Visual elements
        "primary_color": meta.get("primary_color"),
        "font_hierarchy": json.loads(meta["font_hierarchy"]) if meta.get("font_hierarchy") else None,
        "component_spacing": meta.get("component_spacing"),
        # Component details
        "button_types": meta.get("button_types", "").split(",") if meta.get("button_types") else None,
        "form_fields_per_row": meta.get("form_fields_per_row"),
        "table_column_count": meta.get("table_column_count"),
        "table_page_size": meta.get("table_page_size"),
        "popup_types": meta.get("popup_types", "").split(",") if meta.get("popup_types") else None,
        "filter_default_fields": meta.get("filter_default_fields"),
        "filter_total_fields": meta.get("filter_total_fields"),
        "stat_card_count": meta.get("stat_card_count"),
        "component_library": meta.get("component_library"),
        # Data format
        "currency_format": meta.get("currency_format"),
        "date_format": meta.get("date_format"),
        # Validation
        "validation_type": meta.get("validation_type"),
        "validation_rules": json.loads(meta["validation_rules"]) if meta.get("validation_rules") else None,
        # Analysis results
        "full_content": meta.get("full_content"),
        "extracted_specs": json.loads(meta["extracted_specs"]) if meta.get("extracted_specs") else None,
        "unmeasurable_specs": meta.get("unmeasurable_specs", "").split(",") if meta.get("unmeasurable_specs") else None,
    }


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
        """存储原型描述（防重复：基于 id 或 document_id + page_name）"""
        from langchain_core.documents import Document

        metadata = _build_metadata(description)

        # 先通过 id 检查是否已存在
        existing_by_id = self.chroma_store._collection.get(
            where={"id": description.id}
        )

        if existing_by_id and existing_by_id.get("ids"):
            # id 已存在，更新
            doc_id = existing_by_id["ids"][0]
            metadata["id"] = doc_id
            self.chroma_store._collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[description.query_text],
                metadatas=[metadata],
            )
            return doc_id

        # 检查是否已存在（基于 document_id + page_name，仅当两者都非空时）
        existing = None
        if description.document_id and description.page_name:
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
            metadata["id"] = doc_id
            self.chroma_store._collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[description.query_text],
                metadatas=[metadata],
            )
            return doc_id

        # 新增
        doc_id = description.id or str(uuid.uuid4())
        metadata["id"] = doc_id

        doc = Document(
            page_content=description.query_text,
            metadata=metadata,
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

    def update_enriched(self, prototype_id: str, enriched_data: dict) -> bool:
        """更新原型图的丰富元数据（在审查阶段分析后调用）"""
        try:
            existing = self.chroma_store._collection.get(
                where={"id": prototype_id}
            )

            if not existing or not existing.get("ids"):
                return False

            # 获取现有 metadata
            meta = existing["metadatas"][0] if existing.get("metadatas") else {}

            # 更新 enriched 字段
            for key, value in enriched_data.items():
                if value is None:
                    continue
                if isinstance(value, list):
                    value = ",".join(value)
                elif isinstance(value, dict):
                    value = json.dumps(value)
                meta[key] = value

            # 如果有 full_content，更新文档内容
            document = meta.get("query_text", "")
            if "full_content" in enriched_data and enriched_data["full_content"]:
                document = enriched_data["full_content"]

            self.chroma_store._collection.update(
                ids=[prototype_id],
                documents=[document],
                metadatas=[meta],
            )
            return True
        except Exception as e:
            print(f"Error updating prototype enriched data: {e}")
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
            if document_id is not None and meta.get("document_id") != document_id:
                continue

            descriptions.append(PrototypeDescription(**_parse_metadata(meta)))

        return descriptions
