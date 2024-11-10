try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    TELEGRAM_TOKEN: str
    # Use full resource name instead of URL
    VECTOR_SEARCH_ENDPOINT: str = "projects/212717342587/locations/us-central1/indexEndpoints/147947013"
    PROJECT_ID: str = "nomads-laws"
    LOCATION: str = "us-central1"
    
    DEFAULT_LANGUAGE: str = "ru"
    DEFAULT_COUNTRY: str = "georgia"
    DEFAULT_LAW_TYPE: str = "tax"
    
    EMBEDDING_MODEL: str = "text-multilingual-embedding-002" 
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    DIMENSION_SIZE: int = 768
    
    CLOUD_RUN_URL: str = "localhost:8080"

    class Config:
        env_file = ".env"

settings = Settings()