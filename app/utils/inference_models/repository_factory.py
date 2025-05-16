"""Factory for creating model repositories based on environment configuration."""

import os

from app.core.config import config
from app.utils.inference_models.model_repository import (
    ModelRepository,
    LocalCacheRepository,
    S3Repository,
    CachingModelRepository
)


def parse_bool_env(var_name: str, default: bool = False) -> bool:
    """Parse boolean environment variable."""
    val = os.getenv(var_name, str(default).lower())
    return val.lower() in ('true', 'yes', '1', 't', 'y')


def create_model_repository() -> ModelRepository:
    """
    Create the appropriate model repository based on environment settings.

    Uses the following environment configuration:
    - USE_LOCAL_MODEL_REPO: If 'true', only a LocalCacheRepository is used
    - MODEL_REPOSITORY_TYPE: One of 'local', 's3', or 'caching' (default: 'caching')
    - TESTING: If 'true', uses test-friendly repositories (e.g., in-memory when possible)

    Returns:
        The configured model repository
    """
    use_local_only = parse_bool_env('USE_LOCAL_MODEL_REPO')
    testing_mode = parse_bool_env('TESTING')
    repository_type = os.getenv('MODEL_REPOSITORY_TYPE', 'caching').lower()

    # For unit tests, we want to avoid external dependencies
    if use_local_only or testing_mode:
        # Use a base directory that can be overridden in tests
        test_models_dir = os.getenv('TEST_MODELS_DIR', config.MODELS_DIR)
        return LocalCacheRepository(base_dir=test_models_dir)

    # For normal operation, respect the configured repository type
    if repository_type == 'local':
        return LocalCacheRepository()
    elif repository_type == 's3':
        return S3Repository()
    else:  # Default to caching repository
        return CachingModelRepository()
