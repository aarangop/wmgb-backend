
import os
import pytest
import tensorflow as tf

from app.utils.inference_models.model_loader import S3ModelLoader


@pytest.mark.integration
class TestS3ModelLoader_Integration:

    def test_s3_model_loader_init(self):
        """Test S3ModelLoader initialization"""

        # Act
        loader = S3ModelLoader()

        # Assert
        assert loader._client is not None

    def test_s3_model_loader_loads_model_version(self):

        # Arrange
        loader = S3ModelLoader()

        # Act
        loader.load("cat-dog-other-classifier", version="v1")

        assert len(loader.models) > 0
        assert 'cat-dog-other-classifier' in loader.models
        assert isinstance(
            loader.models['cat-dog-other-classifier'], tf.keras.models.Model)

    def test_s3_model_loader_loads_model_latest(self):
        # Arrange
        loader = S3ModelLoader()

        # Act
        loader.load("cat-dog-other-classifier", version="latest")

        assert len(loader.models) > 0
        assert 'cat-dog-other-classifier' in loader.models
        assert isinstance(
            loader.models['cat-dog-other-classifier'], tf.keras.models.Model)

    def test_s3_model_loader_invalid_model(self):
        # Arrange
        loader = S3ModelLoader()

        # Act & Assert
        with pytest.raises(FileNotFoundError, match=f"Model 'non-existing-model' not found in S3"):
            loader.load("non-existing-model")

    def test_s3_model_loader_stores_model_in_local_cache(self, tmp_path):
        # Arrange
        loader = S3ModelLoader()
        model_filename = "cat-dog-other-classifier"
        version = "latest"
        local_cache_dir = tmp_path / "s3_model_cache"
        local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Act
        loader.load(model_filename, version=version,
                    local_cache_dir=str(local_cache_dir))

        # Assert
        assert len(loader.models) > 0
        assert model_filename in loader.models
        assert isinstance(loader.models[model_filename], tf.keras.models.Model)
        # Check if the model is stored in the local cache
        cached_model_path = os.path.join(
            str(local_cache_dir), f"{model_filename}", "v1", "model.h5")
        assert os.path.exists(cached_model_path)
