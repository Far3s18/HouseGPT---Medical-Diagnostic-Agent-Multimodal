from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', env_file_encoding='utf-8')

    TOGETHER_API_KEY: str

    QDRANT_API_KEY: str | None
    QDRANT_URL: str
    QDRANT_PORT: str = "6333"
    QDRANT_HOST: str | None = None

    OPENROUTER_URL: str
    OPENROUTER_API_KEY: str

    LARGE_TEXT_MODEL_NAME: str = "openai/gpt-4o-mini"
    SMALL_TEXT_MODEL_NAME: str = "openai/gpt-4o-mini"
    EMBEDDING_MODEL_NAME: str = "qwen3-embedding:8b"
    STT_MODEL_NAME: str = "medium"
    TTS_MODEL_NAME: str = "eleven_flash_v2_5"
    TTI_MODEL_NAME: str = "black-forest-labs/FLUX.1-schnell-Free"
    ITT_MODEL_NAME: str = "qwen3-vl:8b"

    MEMORY_TOP_K: int = 3
    ROUTER_MESSAGES_TO_ANALYZE: int = 3
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = 20
    TOTAL_MESSAGES_AFTER_SUMMARY: int = 5

    SHORT_TERM_MEMORY_DB_PATH: str = "data/checkpoints.db"

settings = Settings()