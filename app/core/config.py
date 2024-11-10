from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    TELEGRAM_TOKEN: str
    VECTOR_SEARCH_ENDPOINT: str
    
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