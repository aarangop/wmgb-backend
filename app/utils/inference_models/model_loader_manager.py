import os

from loguru import logger
from app.core.config import config
from app.utils.inference_models.model_loader import LocalModelLoader, S3ModelLoader


class ModelLoaderManager:

    _loader = None

    @classmethod
    def get_loader(cls):

        model_source = config.MODEL_SOURCE.lower()
        logger.info(f"Getting model loader for model source '{model_source}'.")
        if cls._loader is not None:
            return cls._loader

        if model_source == 'local':
            models_dir = config.MODELS_DIR
            cls._loader = LocalModelLoader(models_dir)
            return cls._loader

        if model_source == 's3':
            cls._loader = S3ModelLoader()
            return cls._loader
        else:
            raise NotImplementedError(
                f"Loader for {model_source} not implemented"
            )
