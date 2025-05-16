import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
from loguru import logger

from app.core.config import config
from app.utils.aws.s3.client import create_s3_client


class ModelRepository(ABC):
    """Interface for model storage repositories."""

    @abstractmethod
    def get_model(self, model_name: str, version: str = "latest") -> tf.keras.Model:  # type: ignore
        """Retrieve a model from the repository."""
        pass

    @abstractmethod
    def get_available_versions(self, model_name: str) -> List[str]:
        """List available versions for a specific model."""
        pass

    @abstractmethod
    def has_model(self, model_name: str, version: str = "latest") -> bool:
        """Check if repository has the specified model version."""
        pass


class LocalCacheRepository(ModelRepository):
    """Repository for locally stored models."""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize with a base directory for model storage."""
        self.base_dir = base_dir or config.MODELS_DIR
        self.models_cache: Dict[str, tf.keras.Model] = {}  # type: ignore

        logger.info(f"Initialized local cache repository at {self.base_dir}")

        # Create base directory if it doesn't exist
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _get_model_path(self, model_name: str, version: str) -> str:
        """Get the full path to a model file."""
        if version == "latest":
            versions = self.get_available_versions(model_name)
            if not versions:
                return ""
            version = versions[-1]  # Last version is latest

        model_dir = os.path.join(self.base_dir, model_name, version)
        model_path = os.path.join(model_dir, "model.h5")

        return model_path if os.path.exists(model_path) else ""

    def get_available_versions(self, model_name: str) -> List[str]:
        """Get available model versions in local storage."""
        model_dir = os.path.join(self.base_dir, model_name)

        if not os.path.exists(model_dir):
            return []

        # List directories that match version pattern (v1, v2, etc.)
        versions = [d for d in os.listdir(model_dir)
                    if os.path.isdir(os.path.join(model_dir, d)) and
                    re.match(r"v\d+", d)]

        # Sort versions numerically
        versions.sort(key=lambda v: int(v[1:]))
        return versions

    def has_model(self, model_name: str, version: str = "latest") -> bool:
        """Check if model exists in local storage."""
        return bool(self._get_model_path(model_name, version))

    def get_model(self, model_name: str, version: str = "latest") -> tf.keras.Model:  # type: ignore
        """Get model from local storage."""
        # Generate cache key
        if version == "latest":
            versions = self.get_available_versions(model_name)
            if not versions:
                raise FileNotFoundError(
                    f"No versions found for model '{model_name}'")
            actual_version = versions[-1]
        else:
            actual_version = version

        cache_key = f"{model_name}:{actual_version}"

        # Return from memory cache if available
        if cache_key in self.models_cache:
            logger.debug(f"Returning cached model {cache_key} from memory")
            return self.models_cache[cache_key]

        # Get model path
        model_path = self._get_model_path(model_name, version)
        if not model_path:
            raise FileNotFoundError(
                f"Model '{model_name}' version '{version}' not found in local storage"
            )

        # Load model
        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)  # type: ignore

        # Cache in memory
        self.models_cache[cache_key] = model
        return model

    def save_model(
        self,
        model_name: str,
        version: str,
        model: tf.keras.Model  # type: ignore
    ) -> str:
        """Save a model to local storage and return the path."""
        model_dir = os.path.join(self.base_dir, model_name, version)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, "model.h5")
        model.save(model_path)

        # Update memory cache
        cache_key = f"{model_name}:{version}"
        self.models_cache[cache_key] = model

        logger.info(f"Saved model to {model_path}")
        return model_path


# Step 3: Implement the S3Repository
class S3Repository(ModelRepository):
    """Repository for S3-stored models."""

    def __init__(self, bucket_name: Optional[str] = None):
        """Initialize with S3 bucket name."""
        self._client = create_s3_client()
        self._bucket = bucket_name or config.S3_MODELS_BUCKET
        self._env = config.ENV

        logger.info(
            f"Initialized S3 repository with bucket {self._bucket} (env: {self._env})")

    def _parse_version_from_path(self, path: str) -> Optional[int]:
        """Extract version number from S3 path."""
        version_match = re.search(r'/v(\d+)/', path)
        return int(version_match.group(1)) if version_match else None

    def _list_model_objects(self, model_name: str) -> List[dict]:
        """List all objects for a specific model."""
        prefix = f"{self._env}/{model_name}"

        try:
            response = self._client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=prefix
            )

            if 'Contents' not in response:
                logger.warning(
                    f"No objects found with prefix '{prefix}' in bucket '{self._bucket}'")
                return []

            return response['Contents']

        except Exception as e:
            logger.error(f"Error listing objects in S3: {e}")
            raise

    def get_available_versions(self, model_name: str) -> List[str]:
        """Get available model versions in S3."""
        objects = self._list_model_objects(model_name)

        # Extract h5 files and their versions
        model_files = [obj for obj in objects if obj['Key'].endswith('.h5')]

        # Extract versions
        versions = set()
        for obj in model_files:
            version_num = self._parse_version_from_path(obj['Key'])
            if version_num is not None:
                versions.add(f"v{version_num}")

        # Return sorted versions
        return sorted(list(versions), key=lambda v: int(v[1:]))

    def has_model(self, model_name: str, version: str = "latest") -> bool:
        """Check if model exists in S3."""
        versions = self.get_available_versions(model_name)

        if not versions:
            return False

        if version == "latest":
            return True

        return version in versions

    def _get_model_key(self, model_name: str, version: str) -> str:
        """Get the S3 key for a specific model version."""
        objects = self._list_model_objects(model_name)

        # Get version number
        if version == "latest":
            versions = self.get_available_versions(model_name)
            if not versions:
                raise FileNotFoundError(
                    f"No versions found for model '{model_name}' in S3 bucket '{self._bucket}'"
                )
            version = versions[-1]

        version_num = int(version[1:])

        # Find matching file
        model_files = [obj['Key'] for obj in objects
                       if obj['Key'].endswith('.h5') and
                       self._parse_version_from_path(obj['Key']) == version_num]

        if not model_files:
            raise FileNotFoundError(
                f"Model '{model_name}' version '{version}' not found in S3 bucket '{self._bucket}'"
            )

        return model_files[0]

    def get_model(
        self,
            model_name: str,
            version: str = "latest"
    ) -> tf.keras.Model:  # type: ignore
        """
        Get model from S3 - requires downloading to a temporary location.
        Note: This returns the model but doesn't cache it locally.
        Use CachingModelRepository for that functionality.
        """
        import tempfile

        # Get S3 key
        s3_key = self._get_model_key(model_name, version)

        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "model.h5")

        # Download model
        logger.info(
            f"Downloading model from s3://{self._bucket}/{s3_key} to {temp_file}")
        self._client.download_file(
            Bucket=self._bucket,
            Key=s3_key,
            Filename=temp_file
        )

        # Load model
        model = tf.keras.models.load_model(temp_file)  # type: ignore

        # Clean up temp file
        os.unlink(temp_file)
        os.rmdir(temp_dir)

        return model


# Step 4: Implement the CachingModelRepository
class CachingModelRepository(ModelRepository):
    """
    Repository that combines local and S3 storage with caching strategy.
    Tries local first, falls back to S3 and caches locally.
    """

    def __init__(
        self,
        local_repository: Optional[LocalCacheRepository] = None,
        remote_repository: Optional[S3Repository] = None
    ):
        """Initialize with local and remote repositories."""
        self.local_repo = local_repository or LocalCacheRepository()
        self.remote_repo = remote_repository or S3Repository()

        logger.info("Initialized caching model repository")

    def get_available_versions(self, model_name: str) -> List[str]:
        """Get available model versions from both repositories."""
        local_versions = set(
            self.local_repo.get_available_versions(model_name))

        try:
            remote_versions = set(
                self.remote_repo.get_available_versions(model_name))
        except Exception as e:
            logger.warning(f"Error fetching remote versions: {e}")
            remote_versions = set()

        # Combine and sort versions
        all_versions = sorted(
            list(local_versions.union(remote_versions)),
            key=lambda v: int(v[1:])
        )

        return all_versions

    def has_model(self, model_name: str, version: str = "latest") -> bool:
        """Check if model is available in either repository."""
        return (self.local_repo.has_model(model_name, version) or
                self.remote_repo.has_model(model_name, version))

    def get_model(
        self,
            model_name: str,
            version: str = "latest"
    ) -> tf.keras.Model:  # type: ignore
        """
        Get model with caching strategy:
        1. Try local repository first
        2. If not found, get from remote repository 
        3. Cache the model locally
        4. Return the model
        """
        # Resolve actual version if "latest"
        if version == "latest":
            all_versions = self.get_available_versions(model_name)
            if not all_versions:
                raise FileNotFoundError(
                    f"No versions found for model '{model_name}'")
            actual_version = all_versions[-1]
        else:
            actual_version = version

        logger.info(
            f"Retrieving model '{model_name}' version '{actual_version}'")

        # Try local first
        try:
            return self.local_repo.get_model(model_name, actual_version)
        except FileNotFoundError:
            logger.info(f"Model not found in local cache, fetching from S3")

        # Get from remote
        model = self.remote_repo.get_model(model_name, actual_version)

        # Cache locally
        try:
            self.local_repo.save_model(model_name, actual_version, model)
            logger.info(
                f"Cached model '{model_name}:{actual_version}' to local repository")
        except Exception as e:
            logger.warning(f"Failed to cache model locally: {e}")

        return model


# Step 5: Factory function for easy instantiation
def create_model_repository() -> ModelRepository:
    """Create appropriate model repository based on environment."""
    return CachingModelRepository()
