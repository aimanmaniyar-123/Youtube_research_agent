"""
FAISS Vector Database Service – FINAL UPDATED VERSION
Compatible with:
✔ New MasterOrchestrator
✔ New Embedding Service
✔ New Logging System
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from utils_logger import get_logger
from services_embedding_service import get_embedding_service

logger = get_logger("vector_db")


# ---------------------------------------------------------
# Data Class
# ---------------------------------------------------------
@dataclass
class VectorDocument:
    id: str
    text: str
    metadata: Dict[str, Any]
    vector: Optional[np.ndarray] = None


# ---------------------------------------------------------
# FAISS Vector Service
# ---------------------------------------------------------
class FAISSVectorService:
    def __init__(self, config: Dict[str, Any]):
        """
        config = {
            "dimension": 384,
            "faiss_index_path": "./data/faiss_index"
        }
        """
        self.config = config
        self.dimension = config.get("dimension", 384)
        self.index_path = config.get("faiss_index_path", "./data/faiss_index")

        self.index = None
        self.documents: Dict[str, Dict] = {}

        self.initialize()

    # ---------------------------------------------------------
    def initialize(self):
        """Initialize or load FAISS index."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

            if (
                os.path.exists(f"{self.index_path}.faiss")
                and os.path.exists(f"{self.index_path}.pkl")
            ):
                self.load_index()
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info(f"Created new FAISS index with dimension {self.dimension}")

        except Exception as e:
            logger.error(f"FAISS initialize failed: {e}")
            raise

    # ---------------------------------------------------------
    async def reset(self):
        """Clear the FAISS index completely."""
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = {}

            # Delete old files
            if os.path.exists(f"{self.index_path}.faiss"):
                os.remove(f"{self.index_path}.faiss")
            if os.path.exists(f"{self.index_path}.pkl"):
                os.remove(f"{self.index_path}.pkl")

            logger.info("FAISS index reset complete")

        except Exception as e:
            logger.error(f"Failed to reset FAISS index: {e}")
            raise

    # ---------------------------------------------------------
    async def add_documents(self, raw_docs: List[Dict[str, Any]]):
        """
        Accepts raw transcript chunks:
        [
            { "text": "...", "metadata": {...} }
        ]
        Builds embeddings internally.
        """

        try:
            embedder = get_embedding_service()

            # Generate vectors
            texts = [d["text"] for d in raw_docs]
            vectors = await embedder.embed_texts(texts)
            vectors = np.array(vectors).astype("float32")

            faiss.normalize_L2(vectors)

            start_idx = self.index.ntotal
            self.index.add(vectors)

            # Save metadata
            for i, raw in enumerate(raw_docs):
                idx = start_idx + i
                self.documents[str(idx)] = {
                    "id": f"doc_{idx}",
                    "text": raw["text"],
                    "metadata": raw["metadata"]
                }

            # Persist to disk
            self.save_index()
            logger.info(f"FAISS stored {len(raw_docs)} documents")

            return [f"doc_{start_idx + i}" for i in range(len(raw_docs))]

        except Exception as e:
            logger.error(f"FAISS add_documents failed: {e}")
            raise

    # ---------------------------------------------------------
    async def search(self, query: str, k: int = 5):
        """Embed query text and perform FAISS similarity search."""
        try:
            embedder = get_embedding_service()

            # MUST await embedding
            q_vec = await embedder.embed_text(query)

            # Convert to numpy
            q_vec = np.array(q_vec).reshape(1, -1).astype("float32")
            faiss.normalize_L2(q_vec)

            scores, indices = self.index.search(q_vec, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if str(idx) in self.documents:
                    doc = self.documents[str(idx)]
                    results.append({
                        "id": doc["id"],
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                        "score": float(score),
                    })

            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    # ---------------------------------------------------------
    def save_index(self):
        try:
            faiss.write_index(self.index, f"{self.index_path}.faiss")
            with open(f"{self.index_path}.pkl", "wb") as f:
                pickle.dump(self.documents, f)

            logger.info("FAISS index saved")

        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    # ---------------------------------------------------------
    def load_index(self):
        try:
            self.index = faiss.read_index(f"{self.index_path}.faiss")

            with open(f"{self.index_path}.pkl", "rb") as f:
                self.documents = pickle.load(f)

            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise


# ---------------------------------------------------------
# Singleton
# ---------------------------------------------------------
vector_service: Optional[FAISSVectorService] = None


async def create_vector_db_service(config: Dict[str, Any]):
    global vector_service
    if vector_service is None:
        vector_service = FAISSVectorService(config)
    return vector_service


def get_vector_db_service():
    return vector_service


# Export alias for orchestrator
VectorDatabaseService = FAISSVectorService
