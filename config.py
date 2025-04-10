from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    database_url: str
    frontend_origin: str
    face_storage_dir: Path

    class Config:
        env_file = "properties.env"

settings = Settings()
