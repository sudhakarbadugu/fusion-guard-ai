from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "FusionGuard AI"
    VERSION: str = "1.0.0"
    
    # Model settings
    FACE_RECOGNITION_THRESHOLD: float = 0.5
    EMBEDDINGS_DIR: str = "data/embeddings"
    
    # Default Policy Rules if none provided in request
    DEFAULT_ALLOWED_ACTIVITIES: list[str] = [
        "person walking",
        "person working on laptop",
        "person talking on phone"
    ]
    DEFAULT_UNAUTHORIZED_ACTIVITIES: list[str] = [
        "person fighting",
        "person carrying a weapon",
        "person stealing",
        "person breaking things"
    ]

    class Config:
        env_file = ".env"

settings = Settings()
