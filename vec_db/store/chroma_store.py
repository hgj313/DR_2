
import logging

import chromadb
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings

from domain.chunk import Chunk
from langchain_core.documents import Document as LangChainDocument


logger = logging.getLogger(__name__)


class ChromaStore:
    def __init__(
        self,
        collection_name: str = "design_references",
        persist_directory: str = "./chroma_db",
        use_remote: bool = True,
        remote_host: str = "localhost",
        remote_port: int = 8000,
    ):
        """初始化chromaDB 客户端和集合对象

        Args:
            collection_name: 使用的集合名称.
            persist_directory: 本地持久化存储路径（仅 use_remote=False 时使用）.
            use_remote: 是否使用远程服务器（连接 Docker ChromaDB Server）.
            remote_host: 远程服务器地址（仅 use_remote=True 时使用）.
            remote_port: 远程服务器端口（仅 use_remote=True 时使用）.
        """
        self.collection_name = collection_name
        
        if use_remote:
            self._client = chromadb.HttpClient(
                host=remote_host,
                port=remote_port,
                ssl=False,
                settings=Settings(),
                tenant=DEFAULT_TENANT,
                database=DEFAULT_DATABASE,
            )
            logger.info(f"ChromaStore connected to remote: {remote_host}:{remote_port}")
        else:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(),
                tenant=DEFAULT_TENANT,
                database=DEFAULT_DATABASE,
            )
            logger.info(f"ChromaStore initialized locally: path={persist_directory}")
        
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def check_by_content_hash(self, content_hash: str) -> dict | None:
        """通过内容哈希检查chunk是否存在
        
        Args:
            content_hash: 内容哈希值
        
        Returns:
            存在的chunk信息（包含id和metadata），不存在返回None
        """
        try:
            results = self._collection.get(
                where={"content_hash": content_hash},
                include=["metadatas", "documents"]
            )
            
            if results and results.get("ids") and len(results["ids"]) > 0:
                return {
                    "id": results["ids"][0],
                    "metadata": results["metadatas"][0] if results.get("metadatas") else {},
                    "content": results["documents"][0] if results.get("documents") else ""
                }
            return None
        except Exception as e:
            logger.warning(f"Error checking content hash: {e}")
            return None
    
    def get_all_content_hashes(self) -> set[str]:
        """获取所有已存储的内容哈希
        
        Returns:
            所有content_hash的集合
        """
        try:
            results = self._collection.get(include=["metadatas"])
            hashes = set()
            
            if results and results.get("metadatas"):
                for metadata in results["metadatas"]:
                    if metadata and "content_hash" in metadata:
                        hashes.add(metadata["content_hash"])
            
            return hashes
        except Exception as e:
            logger.warning(f"Error getting content hashes: {e}")
            return set()
    
    def store_chunk(self, chunk: Chunk) -> None:
        """持久化存储一个文档的分块（保留完整元数据）
        
        Args:
            chunk: Chunk实体，包含完整的ChunkMetadata
        """
        if chunk.embedding is None:
            raise ValueError(f"Chunk {chunk.id} has no embedding")
        
        metadata_dict = chunk.metadata.to_dict()
        
        self._collection.upsert(
            ids=[str(chunk.id)],
            embeddings=[chunk.embedding],
            documents=[chunk.content],
            metadatas=[metadata_dict],
        )
        logger.debug(f"Stored chunk {chunk.id} with full metadata")

    def store(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """通用存储方法

        Args:
            ids: 文档 ID 列表
            documents: 文档内容列表
            embeddings: 向量列表
            metadatas: 元数据列表
        """
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        embedding: list[float] | list[list[float]],
        k: int = 5
    ) -> list[tuple[LangChainDocument, float]] | list[list[tuple[LangChainDocument, float]]]:
        """查找与查询嵌入向量最相似的文档

        Args:
            embedding: 查询嵌入向量（单向量）或向量列表（批量查询）.
            k: 返回的结果数量.

        Returns:
            单向量查询: 文档-相似元组列表.
            多向量查询: 文档-相似元组列表的列表.
        """
        embeddings = embedding if isinstance(embedding[0], list) else [embedding]
        is_single_query = not isinstance(embedding[0], list)

        if not embeddings:
            return [] if is_single_query else [[]]

        results = self._collection.query(
            query_embeddings=embeddings,
            n_results=k,
        )

        num_queries = len(results.get("ids", []))
        all_results = []

        for query_idx in range(num_queries):
            query_results = []
            result_ids = results["ids"][query_idx]
            result_count = len(result_ids) if result_ids else 0

            for i in range(result_count):
                doc_content = results["documents"][query_idx][i] if results["documents"] and i < len(results["documents"][query_idx]) else ""
                doc_metadata = results["metadatas"][query_idx][i] if results["metadatas"] and i < len(results["metadatas"][query_idx]) else {}
                
                doc = LangChainDocument(
                    page_content=doc_content or "",
                    metadata=doc_metadata or {},
                )
                distance = results["distances"][query_idx][i] if results["distances"] and i < len(results["distances"][query_idx]) else 0.0
                score = 1.0 - distance
                query_results.append((doc, score))

            all_results.append(query_results)

        return all_results[0] if is_single_query else all_results

