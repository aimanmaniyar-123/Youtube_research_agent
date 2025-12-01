"""
Embedding Service - SentenceTransformers
Supports BOTH:
- embed_text(text)        → single string
- embed_texts([text,...]) → list of strings
"""

import asyncio
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from utils_logger import get_logger

logger = get_logger("embedding_service")

_embedding_service_instance = None


class EmbeddingService:
    def __init__(self, model_name: str):
        logger.info(f"🔤 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device="cpu")

    # -------------------------------------------------------
    # SINGLE STRING
    # -------------------------------------------------------
    async def embed_text(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.model.encode, text)

    # -------------------------------------------------------
    # BATCH (LIST OF STRINGS)
    # -------------------------------------------------------
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Required by FAISS.
        Safely handles empty or small lists.
        """
        if not texts:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.model.encode, texts)


# -------------------------------------------------------
# Accessor
# -------------------------------------------------------
def get_embedding_service(config: Optional[dict] = None):
    global _embedding_service_instance

    if _embedding_service_instance is None:
        if config is None:
            raise ValueError("EmbeddingService requires config with 'model' on first load.")

        model_name = config.get("model")
        _embedding_service_instance = EmbeddingService(model_name)

    return _embedding_service_instance
