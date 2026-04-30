import logging


from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BgeM3Embeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = None,
        normalize_embeddings: bool = True,
    ):
        if device is None:
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize_embeddings = normalize_embeddings
        logger.info(f"Model {model_name} loaded on device: {device}")

    def embed_query(self, text_query: str) -> list[float]:
        """
        嵌入查询文本
        """
        embedding = self.model.encode(
            text_query,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        嵌入文档列表
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            batch_size=32,
        )
        return embeddings.tolist()
