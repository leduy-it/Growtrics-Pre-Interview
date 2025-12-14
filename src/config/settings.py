"""Application settings and configuration."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5.2"
    
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash-latest"
    
    OUTPUT_DIR: str = "outputs"
    LOG_LEVEL: str = "INFO"
    
    OPENAI_INPUT_COST_PER_1M: float = 1.75
    OPENAI_OUTPUT_COST_PER_1M: float = 14.00
    
    GEMINI_INPUT_COST_PER_1M: float = 0.30
    GEMINI_OUTPUT_COST_PER_1M: float = 2.50
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

