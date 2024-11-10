from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    # API Keys & Endpoints
    GOOGLE_API_KEY: str
    TELEGRAM_TOKEN: str
    VECTOR_SEARCH_ENDPOINT: str
    
    # Defaults
    DEFAULT_LANGUAGE: str = "ru"
    DEFAULT_COUNTRY: str = "georgia"
    DEFAULT_LAW_TYPE: str = "tax"
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "models/text-multilingual-embedding-002"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    DIMENSION_SIZE: int = 768

    # Cloud Run URL (without https://)
    CLOUD_RUN_URL: str
    
    # Webhook settings
    WEBHOOK_PATH: str = "/telegram-webhook"
    
    @property
    def webhook_url(self) -> str:
        return f"https://{self.CLOUD_RUN_URL}{self.WEBHOOK_PATH}"

    class Config:
        env_file = ".env"

settings = Settings()