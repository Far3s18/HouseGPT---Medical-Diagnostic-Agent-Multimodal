from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENROUTER_URL: str
    OPENROUTER_API_KEY: str

    GROQ_API_KEY: str

    SMALL_TEXT_MODEL_NAME: str
    LARGE_TEXT_MODEL_NAME: str
    ITT_MODEL_NAME: str

    STT_MODEL_NAME: str = "whisper-large-v3"

    EMBEDDING_MODEL_NAME: str

    POSTGRES_URI: str
    POSTGRES_MIN_CONNECTIONS: int = 2
    POSTGRES_MAX_CONNECTIONS: int = 10

    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None

    MEMORY_TOP_K: int = 5
    SHORT_TERM_MEMORY_DB_PATH: str = "./data/memory.db"

    ROUTER_MESSAGES_TO_ANALYZE: int = 5
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = 20
    TOTAL_MESSAGES_AFTER_SUMMARY: int = 6


settings = Settings()