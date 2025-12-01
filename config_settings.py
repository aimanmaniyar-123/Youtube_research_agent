"""
Application Settings Configuration - Groq + FAISS
"""
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # Application
    app_name: str = "YouTube Research Agent"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = ["*"]

    # API Keys
    youtube_api_key: str
    youtube_quota_per_day: int = 10000
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"  # Fast, high-quality model
    default_llm_provider: str = "groq"
    redis_url: str = "redis://localhost:6379/0"

    # Database
    database_url: str

    # Vector Database (FAISS)
    vector_db_provider: str = "faiss"
    faiss_index_path: str = "./data/faiss_index"
    vector_dimension: int = 384

    # Embeddings
    embedding_provider: str = "sentence_transformers"
    sentence_transformers_model: str = "all-MiniLM-L6-v2"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    task_timeout: int = 300
    soft_task_timeout: int = 270
    max_retries: int = 3

    # Research Settings
    max_videos_per_channel: int = 100
    chunk_size: int = 512
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
