from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    database_url: str = "sqlite:///./retainiq.db"
    model_dir: str = "./ml/models"
    data_dir: str = "./data"
    chroma_dir: str = "./chroma_db"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()