import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    database_url: str = "postgresql+asyncpg://videocuts:videocuts_dev@localhost:5432/videocuts"
    storage_path: str = "./storage"
    openai_api_key: str = ""
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

@lru_cache
def get_settings() -> Settings:
    return Settings()
