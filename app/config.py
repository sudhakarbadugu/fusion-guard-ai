from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "FusionGuard AI"
    VERSION: str = "1.0.0"
    
    # Model settings
    FACE_RECOGNITION_THRESHOLD: float = 0.5
    EMBEDDINGS_DIR: str = "data/embeddings"
    
    # Default Policy Rules if none provided in request
    DEFAULT_ALLOWED_ACTIVITIES: list[str] = [
        'a person working on a computer',
        'a person writing on a whiteboard',
        'a person reading a book',
        'a group of students discussing',
        'a person conducting an experiment',
    ]
    DEFAULT_UNAUTHORIZED_ACTIVITIES: list[str] = [
        'a person taking photos of equipment',
        'a person eating food in the lab',
        'a person sleeping at the desk',
        'an unauthorized person in the lab',
        'a person tampering with equipment',
    ]

    class Config:
        env_file = ".env"

settings = Settings()
