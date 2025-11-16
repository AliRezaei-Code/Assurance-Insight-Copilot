"""Application configuration helpers."""
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    environment: str = Field(default="local")
    database_url: str = Field(default="sqlite+aiosqlite:///./assurance_insight.db")
    azure_openai_endpoint: str = Field(default="https://example-openai.openai.azure.com")
    azure_openai_api_key: str = Field(default="changeme", repr=False)
    azure_openai_chat_deployment: str = Field(default="gpt-4o-mini")
    azure_openai_embedding_deployment: str = Field(default="text-embedding-3-large")
    azure_ai_search_endpoint: str = Field(default="https://example-search.search.windows.net")
    azure_ai_search_api_key: str = Field(default="changeme", repr=False)
    azure_ai_search_index_name: str = Field(default="documents-index")
    azure_storage_connection_string: str = Field(default="DefaultEndpointsProtocol=https;AccountName=fake;")
    azure_storage_container: str = Field(default="documents")
    azure_content_safety_endpoint: str = Field(default="https://example-safety.cognitiveservices.azure.com")
    azure_content_safety_api_key: str = Field(default="changeme", repr=False)
    jwt_secret_key: str = Field(default="super-secret", repr=False)
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=60)
    rag_top_k: int = Field(default=4)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()


settings = get_settings()
