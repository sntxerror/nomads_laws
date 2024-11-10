try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    TELEGRAM_TOKEN: str
    VECTOR_SEARCH_ENDPOINT: str = "https://147947013.us-central1-212717342587.vdb.vertexai.goog"
    PROJECT_ID: str = "nomads-laws"
    LOCATION: str = "us-central1"
    
    DEFAULT_LANGUAGE: str = "ru"
    DEFAULT_COUNTRY: str = "georgia"
    DEFAULT_LAW_TYPE: str = "tax"
    
    EMBEDDING_MODEL: str = "models/text-multilingual-embedding-002"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    DIMENSION_SIZE: int = 768
    
    CLOUD_RUN_URL: str

    class Config:
        env_file = ".env"

settings = Settings()