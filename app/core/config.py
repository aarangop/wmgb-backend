import os
from typing import List
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # Environment
    ENV: str = os.getenv("env", "development")

    # API Configuration
    API_VERSION: str = os.getenv("API_VERSION", "v1")
    API_PREFIX: str = f"/api/{API_VERSION}"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000", "https://localhost:3000"]
    PORT: int = int(os.getenv("PORT", 8000))

    # S3 Configuration
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-2")
    S3_MODELS_BUCKET: str = os.getenv(
        "S3_MODELS_BUCKET", "whos-my-good-boy-models")

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    JSON_LOGS: bool = os.getenv("JSON_LOGS", "").lower() == "true"

    # Model Configuration
    MODEL_SOURCE: str = os.getenv("MODEL_SOURCE", "local")
    MODELS_DIR: str = os.getenv("MODELS_DIR", "./models")
    CAT_DOG_OTHER_CLASSIFIER: str = os.getenv("CAT_DOG_OTHER_CLASSIFIER", "cat_dog_other_classifier.h5")

    model_config = ConfigDict(
        extra='allow',
        env_file=".env",
        case_sensitive=True
    )


config = Config()
