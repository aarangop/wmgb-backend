import os
import boto3
from loguru import logger
import tensorflow as tf
from abc import ABC, abstractmethod
from app.core.config import config
from app.utils.aws.s3.client import create_s3_client
import re


class ModelLoader(ABC):

    @abstractmethod
    def load(self, model_filename: str):
        """Load the model from a specified path or source."""
        pass

    def is_model_available(self):
        """Check if the model is available for inference."""
        return len(self.models) > 0 and all(isinstance(model, tf.keras.models.Model) for model in self.models.values())


class LocalModelLoader(ModelLoader):
    """
    Load ML models from local storage. Used for development on local machine. 
    """

    def __init__(self, models_dir: str = './models'):
        if not os.path.exists(models_dir):
            raise ValueError(
                f"Models directory '{models_dir}' does not exist.")

        self.models_dir = models_dir
        self.models = {}

        logger.info(
            f"Instantiating local model loader with models directory '{self.models_dir}'"
        )

    def load(self, model_filename: str):
        """Load the model from a local path."""
        logger.info(f"Loading model '{model_filename}' from {self.models_dir}")
        if model_filename in self.models:
            return self.models[model_filename]

        model_path = os.path.join(self.models_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file '{model_path}'")

        model = tf.keras.models.load_model(model_path)
        self.models[model_filename] = model

        return model


class S3ModelLoader(ModelLoader):
    """
    Loader to get download the model from S3.
    """

    def __init__(self):
        """
        Initialize the S3 model loader, setting up an aws s3 client.
        """
        self._client = create_s3_client()
        self._bucket = config.S3_MODELS_BUCKET
        self.models = {}

    def _get_model_version_from_path(self, model_path):
        """
        Extract the model version from the S3 path.
        """
        # The version is stored in the format 'model_name/v1/', 'model_name/v2/', etc.
        # Extract the version part from the path
        version_pattern = r'/v(\d)/+'
        match = re.search(version_pattern, model_path)
        if match:
            return int(match.group(1))
        return None

    def _get_versioned_model_paths(self, bucket_objects):
        """
        Get the paths of all versioned models in the S3 bucket.
        """
        # Contents should return a list of directories where each model version is stored
        # The versions are stored in the format 'model_name/v1/', 'model_name/v2/', etc.
        # Extract the version part from the path

        # Use list comprehension to get a tuple with the version number and the path
        models = [obj['Key']
                  for obj in bucket_objects if obj['Key'].endswith('h5')]

        if not models:
            raise FileNotFoundError(
                f"No versions found for model in S3 bucket '{self._bucket}'")

        versions = [self._get_model_version_from_path(model)
                    for model in models]

        model_versions = zip(versions, models)

        return list(model_versions)

    def _get_latest_model_version_path(self, bucket_objects):

        # Contents should return a list of directories where each model version is stored
        # The versions are stored in the format 'model_name/v1/', 'model_name/v2/', etc.
        # Extract the version part from the path

        model_versions = self._get_versioned_model_paths(bucket_objects)

        # Sort versions and get the latest one
        model_versions.sort(key=lambda x: x[0], reverse=True)

        latest_model = model_versions[0][1]
        logger.info(
            f"Latest model version found: {latest_model} in S3 bucket '{self._bucket}'")
        return latest_model

    def _get_model_version_path(self, bucket_objects, version):
        """
        Get the path of a specific version of the model in the S3 bucket.
        """
        # Check that the version is valid, should be in the format 'v1', 'v2', etc.
        if not re.match(r'v\d+', version):
            raise ValueError(
                f"Invalid version format '{version}'. Expected format is 'v1', 'v2', etc.")

        # Get the version number from the version string
        version_number = int(re.search(r'v(\d)+', version).group(1))

        model_versions = self._get_versioned_model_paths(bucket_objects)

        model = next(
            filter(lambda x: x[0] == version_number, model_versions), None)

        if not model:
            raise FileNotFoundError(
                f"Model version '{version}' not found in S3 bucket '{self._bucket}'")

        return model[1]

    def load(self, model_name, version='latest', local_cache_dir=None):
        """
        Load model from S3 bucket
        """
        logger.info(
            f"Loading model '{model_name}' from S3 bucket '{self._bucket}'")

        if model_name in self.models:
            logger.debug(f"Model '{model_name}' already loaded in memory, returning cached version")
            return self.models[model_name]

        # Check environment variable for model source
        env = config.ENV
        logger.debug(f"Environment: {env}")

        prefix = f"{env}/{model_name}"
        logger.debug(f"Looking for models with prefix: '{prefix}'")

        # Get list of objects in the S3 bucket
        try:
            objects = self._client.list_objects_v2(
                Bucket=self._bucket, Prefix=prefix
            )
            logger.debug(f"S3 response: {objects}")
        except Exception as e:
            logger.error(f"Error listing objects in S3: {e}")
            raise

        if 'Contents' not in objects:
            logger.error(f"No objects found with prefix '{prefix}' in bucket '{self._bucket}'")
            
            # Try listing the bucket contents without prefix to see what's available
            try:
                all_objects = self._client.list_objects_v2(Bucket=self._bucket)
                if 'Contents' in all_objects:
                    keys = [obj['Key'] for obj in all_objects['Contents']]
                    logger.debug(f"Available objects in bucket: {keys}")
            except Exception as e:
                logger.error(f"Error listing all objects in bucket: {e}")
                
            raise FileNotFoundError(
                f"Model '{model_name}' not found in S3 bucket '{self._bucket}'")
            
        logger.debug(f"Found {len(objects['Contents'])} objects with prefix '{prefix}'")

        if version == 'latest':
            model_path = self._get_latest_model_version_path(
                objects['Contents'])
            version = f'v{self._get_model_version_from_path(model_path)}'
            logger.info(
                f"Latest model version '{version}' found in S3 bucket '{self._bucket}'")
        else:
            model_path = self._get_model_version_path(
                objects['Contents'], version)

        # Download the model file from S3
        logger.info(
            f"Downloading model '{model_name}' from S3 bucket '{self._bucket}' to local storage"
        )

        # Check if directory for models exists
        models_dir = local_cache_dir or config.MODELS_DIR

        if local_cache_dir:
            logger.info(
                f"Using custom local cache directory '{local_cache_dir}' for model storage"
            )
        else:
            logger.info(
                f"Using default local cache directory '{config.MODELS_DIR}' for model storage"
            )

        model_dir = os.path.join(models_dir, model_name, version)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_filename = os.path.join(model_dir, "model.h5")

        if not os.path.exists(model_filename):
            self._client.download_file(
                Bucket=self._bucket, Key=model_path, Filename=model_filename
            )
            logger.info(
                f"Model '{model_name}' downloaded from S3 bucket '{self._bucket}' to '{model_filename}'"
            )
        else:
            logger.info(
                f"Model '{model_name}' already exists locally at '{model_filename}'"
            )

        # Load the model
        model = tf.keras.models.load_model(model_filename)
        self.models[model_name] = model
