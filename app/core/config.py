from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    TELEGRAM_TOKEN: str
    DEFAULT_LANGUAGE: str = "ru"
    DEFAULT_COUNTRY: str = "georgia"
    DEFAULT_LAW_TYPE: str = "tax"
    EMBEDDING_MODEL: str = "models/text-multilingual-embedding-002"
    VECTOR_SEARCH_ENDPOINT: str

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
