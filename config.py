"""
Application configuration loaded from environment variables.
"""

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    top_k_tables: int = int(os.getenv("TOP_K_TABLES", "6"))
    beaver_db_split: str = os.getenv("BEAVER_DB_SPLIT", "nova")
    hf_token: str = os.getenv("HF_TOKEN", "")

    class Config:
        env_file = ".env"


settings = Settings()
