import os
from typing import List
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


def parse_bool_env(val: str) -> bool:
    """Parse boolean environment variable."""
    return val.lower() in ('true', 'yes', '1', 't', 'y')


class Config(BaseSettings):
    # Environment
    ENV: str = os.getenv("ENV", "development")
    TESTING: bool = parse_bool_env(os.getenv("TESTING", "false"))

    # API Configuration
    API_VERSION: str = os.getenv("API_VERSION", "v1")
    API_PREFIX: str = f"/api/{API_VERSION}"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000", "https://localhost:3000"]
    PORT: int = int(os.getenv("PORT", "8000"))

    # S3 Configuration
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-2")
    S3_MODELS_BUCKET: str = os.getenv(
        "S3_MODELS_BUCKET", "whos-my-good-boy-models")

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    JSON_LOGS: bool = parse_bool_env(os.getenv("JSON_LOGS", "false"))

    # Model Configuration
    USE_LOCAL_MODEL_REPO: bool = parse_bool_env(
        os.getenv("USE_LOCAL_MODEL_REPO", "false"))
    MODEL_REPOSITORY_TYPE: str = os.getenv("MODEL_REPOSITORY_TYPE", "caching")

    MODELS_DIR: str = os.getenv("MODELS_DIR", "./models")
    TEST_MODELS_DIR: str = os.getenv("TEST_MODELS_DIR", "./models/test")
    CAT_DOG_OTHER_CLASSIFIER: str = os.getenv(
        "CAT_DOG_OTHER_CLASSIFIER", "cat-dog-other-classifier")

    model_config = ConfigDict(
        extra='allow',
        env_file=".env",
        case_sensitive=True
    )  # type: ignore


config = Config()
