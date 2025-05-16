import os
import shutil
import tempfile
import pytest
import tensorflow as tf
from unittest.mock import patch, MagicMock, ANY

from app.utils.inference_models.model_repository import (
    ModelRepository,
    LocalCacheRepository,
    S3Repository,
    CachingModelRepository,
)
from tests.unit.utils.inference_models.create_dummy_model import create_dummy_model


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model storage during tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up temp directory after tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def dummy_model():
    """Create a dummy TensorFlow model for testing."""
    return create_dummy_model()


@pytest.fixture(name="mock_s3_client", autouse=True)
def patch_s3_client():
    """Create a mocked S3 client for testing without actual S3 access.
    This is set to autouse=True to ensure all tests use the mock."""
    with patch('app.utils.inference_models.model_repository.create_s3_client') as mock_client:
        mock = MagicMock()
        mock_client.return_value = mock
        yield mock


@pytest.fixture
def local_repo_with_model(temp_model_dir, dummy_model):
    """Set up a LocalCacheRepository with a model for testing."""
    # Initialize repository with temp directory
    repo = LocalCacheRepository(base_dir=temp_model_dir)

    # Create model structure (v1) and save model
    model_name = "test_model"
    version = "v1"
    model_dir = os.path.join(temp_model_dir, model_name, version)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.h5")
    dummy_model.save(model_path)

    return repo


class TestLocalCacheRepository:

    def test_initialization(self, temp_model_dir):
        """Test that LocalCacheRepository initializes correctly."""
        # Act
        repo = LocalCacheRepository(base_dir=temp_model_dir)

        # Assert
        assert repo.base_dir == temp_model_dir
        assert isinstance(repo.models_cache, dict)
        assert os.path.exists(temp_model_dir)

    def test_get_available_versions_empty(self, temp_model_dir):
        """Test getting versions when none are available."""
        # Arrange
        repo = LocalCacheRepository(base_dir=temp_model_dir)
        model_name = "nonexistent_model"

        # Act
        versions = repo.get_available_versions(model_name)

        # Assert
        assert versions == []

    def test_get_available_versions(self, temp_model_dir):
        """Test getting available versions when models exist."""
        # Arrange
        repo = LocalCacheRepository(base_dir=temp_model_dir)
        model_name = "test_model"

        # Create version directories
        for version in ["v1", "v2", "v3"]:
            os.makedirs(os.path.join(temp_model_dir,
                        model_name, version), exist_ok=True)

        # Add a non-version directory that should be ignored
        os.makedirs(os.path.join(temp_model_dir,
                    model_name, "other_dir"), exist_ok=True)

        # Act
        versions = repo.get_available_versions(model_name)

        # Assert
        assert versions == ["v1", "v2", "v3"]

    def test_has_model_true(self, local_repo_with_model):
        """Test has_model returns True when model exists."""
        # Act & Assert
        assert local_repo_with_model.has_model("test_model", "v1") is True
        assert local_repo_with_model.has_model("test_model", "latest") is True

    def test_has_model_false(self, local_repo_with_model):
        """Test has_model returns False when model doesn't exist."""
        # Act & Assert
        assert local_repo_with_model.has_model("nonexistent_model") is False
        assert local_repo_with_model.has_model("test_model", "v999") is False

    def test_get_model_specific_version(self, local_repo_with_model):
        """Test getting a model with a specific version."""
        # Act
        model = local_repo_with_model.get_model("test_model", "v1")

        # Assert
        assert isinstance(model, tf.keras.models.Model)  # type: ignore
        assert "test_model:v1" in local_repo_with_model.models_cache

        # Second call should use cache
        cached_model = local_repo_with_model.get_model("test_model", "v1")
        assert cached_model is model  # Same object reference due to caching

    def test_get_model_latest_version(self, local_repo_with_model, temp_model_dir, dummy_model):
        """Test getting the latest model version."""
        # Arrange - add a v2 version
        model_name = "test_model"
        v2_dir = os.path.join(temp_model_dir, model_name, "v2")
        os.makedirs(v2_dir, exist_ok=True)
        dummy_model.save(os.path.join(v2_dir, "model.h5"))

        # Act
        model = local_repo_with_model.get_model(model_name, "latest")

        # Assert
        assert isinstance(model, tf.keras.models.Model)  # type: ignore
        assert "test_model:v2" in local_repo_with_model.models_cache

    def test_get_model_nonexistent(self, local_repo_with_model):
        """Test getting a model that doesn't exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            local_repo_with_model.get_model("nonexistent_model")

        with pytest.raises(FileNotFoundError):
            local_repo_with_model.get_model("test_model", "v999")

    def test_save_model(self, temp_model_dir, dummy_model):
        """Test saving a model to the repository."""
        # Arrange
        repo = LocalCacheRepository(base_dir=temp_model_dir)
        model_name = "new_model"
        version = "v1"

        # Act
        saved_path = repo.save_model(model_name, version, dummy_model)

        # Assert
        expected_path = os.path.join(
            temp_model_dir, model_name, version, "model.h5")
        assert saved_path == expected_path
        assert os.path.exists(expected_path)
        assert f"{model_name}:{version}" in repo.models_cache


class TestS3Repository:

    def test_initialization(self, mock_s3_client):
        """Test that S3Repository initializes correctly."""
        # Act
        repo = S3Repository(bucket_name="test-bucket")

        # Assert
        assert repo._bucket == "test-bucket"
        assert repo._client == mock_s3_client

    def test_parse_version_from_path(self):
        """Test parsing versions from S3 paths."""
        # Arrange
        repo = S3Repository(bucket_name="test-bucket")

        # Act & Assert
        assert repo._parse_version_from_path("dev/model/v1/model.h5") == 1
        assert repo._parse_version_from_path("dev/model/v123/model.h5") == 123
        assert repo._parse_version_from_path(
            "dev/model/other/model.h5") is None

    @patch('app.utils.inference_models.model_repository.config')
    def test_list_model_objects(self, mock_config, mock_s3_client):
        """Test listing objects for a specific model."""
        # Arrange
        mock_config.ENV = "dev"  # Set the environment to 'dev'
        repo = S3Repository(bucket_name="test-bucket")
        model_name = "test_model"
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'dev/test_model/v1/model.h5'},
                {'Key': 'dev/test_model/v2/model.h5'},
                {'Key': 'dev/test_model/v2/metadata.json'},
            ]
        }

        # Act
        objects = repo._list_model_objects(model_name)

        # Assert
        assert len(objects) == 3
        mock_s3_client.list_objects_v2.assert_called_with(
            Bucket="test-bucket",
            Prefix="dev/test_model"
        )

    def test_list_model_objects_empty(self, mock_s3_client):
        """Test listing objects when none are available."""
        # Arrange
        repo = S3Repository(bucket_name="test-bucket")
        model_name = "nonexistent_model"

        # Configure the mock to return an empty response for this specific model
        def mock_list_objects_side_effect(**kwargs):
            prefix = kwargs.get('Prefix', '')

            # For any prefix containing 'nonexistent_model', return empty response
            if 'nonexistent_model' in prefix:
                return {}  # No 'Contents' key

            # For other prefixes, return some content (not used in this test)
            return {
                'Contents': [
                    {'Key': 'dev/test_model/v1/model.h5'}
                ]
            }

        # Set the side effect for list_objects_v2
        mock_s3_client.list_objects_v2.side_effect = mock_list_objects_side_effect

        # Act
        objects = repo._list_model_objects(model_name)

        # Assert
        assert objects == []

    def test_get_available_versions(self, mock_s3_client):
        """Test getting available versions from S3."""
        # Arrange
        repo = S3Repository(bucket_name="test-bucket")
        model_name = "test_model"
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'dev/test_model/v1/model.h5'},
                {'Key': 'dev/test_model/v1/metadata.json'},
                {'Key': 'dev/test_model/v3/model.h5'},
                {'Key': 'dev/test_model/v2/model.h5'},
                {'Key': 'dev/test_model/other/file.txt'},
            ]
        }

        # Act
        versions = repo.get_available_versions(model_name)

        # Assert
        assert versions == ["v1", "v2", "v3"]

    def test_has_model(self, mock_s3_client):
        """Test checking if a model exists in S3."""
        # Arrange
        repo = S3Repository(bucket_name="test-bucket")
        model_name = "test_model"

        # Create a side effect function that returns different responses based on the prefix
        def mock_list_objects_side_effect(**kwargs):
            prefix = kwargs.get('Prefix', '')

            # If the prefix contains test_model, return objects
            if 'test_model' in prefix:
                return {
                    'Contents': [
                        {'Key': 'dev/test_model/v1/model.h5'},
                        {'Key': 'dev/test_model/v2/model.h5'},
                        {'Key': 'dev/test_model/v2/metadata.json'},
                    ]
                }
            # For any other prefix, return empty response (no 'Contents' key)
            return {}

        # Set the side effect for list_objects_v2
        mock_s3_client.list_objects_v2.side_effect = mock_list_objects_side_effect

        # Act & Assert
        assert repo.has_model(model_name, "v1") is True
        assert repo.has_model(model_name, "v2") is True
        assert repo.has_model(model_name, "latest") is True
        # Now this should work correctly since our mock responds to different prefixes
        assert repo.has_model("nonexistent_model", "v1") is False

    def test_get_model_key(self, mock_s3_client):
        """Test getting the correct S3 key for a model."""
        # Arrange
        repo = S3Repository(bucket_name="test-bucket")
        model_name = "test_model"

        # Create a side effect function that returns different responses based on the prefix
        def mock_list_objects_side_effect(**kwargs):
            prefix = kwargs.get('Prefix', '')

            # If the prefix contains test_model, return objects
            if 'test_model' in prefix:
                return {
                    'Contents': [
                        {'Key': 'dev/test_model/v1/model.h5'},
                        {'Key': 'dev/test_model/v2/model.h5'},
                    ]
                }
            # For any other prefix, return empty response
            return {}

        # Set the side effect for list_objects_v2
        mock_s3_client.list_objects_v2.side_effect = mock_list_objects_side_effect

        # Mock get_available_versions for the latest check
        with patch.object(repo, 'get_available_versions') as mock_get_versions:
            mock_get_versions.return_value = ["v1", "v2"]

            # Act & Assert - specific version
            assert repo._get_model_key(
                model_name, "v1") == 'dev/test_model/v1/model.h5'
            assert repo._get_model_key(
                model_name, "v2") == 'dev/test_model/v2/model.h5'

            # Act & Assert - latest version
            assert repo._get_model_key(
                model_name, "latest") == 'dev/test_model/v2/model.h5'

    def test_get_model(self, mock_s3_client):
        """Test getting a model from S3."""
        # Arrange
        repo = S3Repository(bucket_name="test-bucket")
        model_name = "test_model"
        version = "v1"

        # Mock the necessary methods and functions
        with patch.object(repo, '_get_model_key', return_value='dev/test_model/v1/model.h5') as mock_get_key, \
                patch('tempfile.mkdtemp', return_value='/tmp/mock_dir') as mock_mkdtemp, \
                patch('os.path.join', return_value='/tmp/mock_dir/model.h5') as mock_join, \
                patch('tensorflow.keras.models.load_model') as mock_load_model, \
                patch('os.unlink') as mock_unlink, \
                patch('os.rmdir') as mock_rmdir:

            # Set up the mock objects
            mock_model = MagicMock(spec=tf.keras.models.Model)  # type: ignore
            mock_load_model.return_value = mock_model

            # Act
            model = repo.get_model(model_name, version)

            # Assert
            mock_get_key.assert_called_with(model_name, version)
            mock_s3_client.download_file.assert_called_with(
                Bucket="test-bucket",
                Key='dev/test_model/v1/model.h5',
                Filename='/tmp/mock_dir/model.h5'
            )
            mock_load_model.assert_called_with('/tmp/mock_dir/model.h5')
            mock_unlink.assert_called_with('/tmp/mock_dir/model.h5')
            mock_rmdir.assert_called_with('/tmp/mock_dir')
            assert model == mock_model


class TestCachingModelRepository:

    def test_initialization(self):
        """Test that CachingModelRepository initializes with correct repositories."""
        # Arrange
        mock_local = MagicMock(spec=LocalCacheRepository)
        mock_remote = MagicMock(spec=S3Repository)

        # Act
        repo = CachingModelRepository(
            local_repository=mock_local,
            remote_repository=mock_remote
        )

        # Assert
        assert repo.local_repo == mock_local
        assert repo.remote_repo == mock_remote

    def test_initialization_defaults(self):
        """Test initialization with default repositories."""
        # Arrange & Act
        with patch('app.utils.inference_models.model_repository.LocalCacheRepository') as mock_local_class, \
                patch('app.utils.inference_models.model_repository.S3Repository') as mock_s3_class:

            mock_local = MagicMock(spec=LocalCacheRepository)
            mock_remote = MagicMock(spec=S3Repository)
            mock_local_class.return_value = mock_local
            mock_s3_class.return_value = mock_remote

            repo = CachingModelRepository()

            # Assert
            assert repo.local_repo == mock_local
            assert repo.remote_repo == mock_remote
            mock_local_class.assert_called_once()
            mock_s3_class.assert_called_once()

    def test_get_available_versions(self):
        """Test getting available versions from both repositories."""
        # Arrange
        mock_local = MagicMock(spec=LocalCacheRepository)
        mock_remote = MagicMock(spec=S3Repository)

        mock_local.get_available_versions.return_value = ["v1", "v2"]
        mock_remote.get_available_versions.return_value = ["v2", "v3"]

        repo = CachingModelRepository(
            local_repository=mock_local,
            remote_repository=mock_remote
        )
        model_name = "test_model"

        # Act
        versions = repo.get_available_versions(model_name)

        # Assert
        assert versions == ["v1", "v2", "v3"]
        mock_local.get_available_versions.assert_called_with(model_name)
        mock_remote.get_available_versions.assert_called_with(model_name)

    def test_get_available_versions_remote_error(self):
        """Test handling when remote repository returns an error."""
        # Arrange
        mock_local = MagicMock(spec=LocalCacheRepository)
        mock_remote = MagicMock(spec=S3Repository)

        mock_local.get_available_versions.return_value = ["v1", "v2"]
        mock_remote.get_available_versions.side_effect = Exception("S3 error")

        repo = CachingModelRepository(
            local_repository=mock_local,
            remote_repository=mock_remote
        )
        model_name = "test_model"

        # Act
        versions = repo.get_available_versions(model_name)

        # Assert
        assert versions == ["v1", "v2"]
        mock_local.get_available_versions.assert_called_with(model_name)
        mock_remote.get_available_versions.assert_called_with(model_name)

    def test_has_model_true(self):
        """Test has_model returns True when model exists in either repository."""
        # Arrange
        mock_local = MagicMock(spec=LocalCacheRepository)
        mock_remote = MagicMock(spec=S3Repository)

        scenarios = [
            # (local, remote, expected)
            (True, True, True),   # Both have it
            (True, False, True),  # Only local has it
            (False, True, True),  # Only remote has it
        ]

        repo = CachingModelRepository(
            local_repository=mock_local,
            remote_repository=mock_remote
        )
        model_name = "test_model"
        version = "v1"

        for local_has, remote_has, expected in scenarios:
            # Reset mocks
            mock_local.reset_mock()
            mock_remote.reset_mock()

            # Set up returns
            mock_local.has_model.return_value = local_has
            mock_remote.has_model.return_value = remote_has

            # Act
            result = repo.has_model(model_name, version)

            # Assert
            assert result == expected
            mock_local.has_model.assert_called_with(model_name, version)

            # Remote should only be called if local doesn't have it
            if not local_has:
                mock_remote.has_model.assert_called_with(model_name, version)

    def test_has_model_false(self):
        """Test has_model returns False when model doesn't exist in either repository."""
        # Arrange
        mock_local = MagicMock(spec=LocalCacheRepository)
        mock_remote = MagicMock(spec=S3Repository)

        mock_local.has_model.return_value = False
        mock_remote.has_model.return_value = False

        repo = CachingModelRepository(
            local_repository=mock_local,
            remote_repository=mock_remote
        )
        model_name = "test_model"
        version = "v1"

        # Act
        result = repo.has_model(model_name, version)

        # Assert
        assert result is False
        mock_local.has_model.assert_called_with(model_name, version)
        mock_remote.has_model.assert_called_with(model_name, version)

    def test_get_model_from_local(self):
        """Test getting a model when it exists in local repository."""
        # Arrange
        mock_local = MagicMock(spec=LocalCacheRepository)
        mock_remote = MagicMock(spec=S3Repository)

        mock_model = MagicMock(spec=tf.keras.Model)
        mock_local.get_model.return_value = mock_model

        repo = CachingModelRepository(
            local_repository=mock_local,
            remote_repository=mock_remote
        )
        model_name = "test_model"
        version = "v1"

        # Act
        model = repo.get_model(model_name, version)

        # Assert
        assert model == mock_model
        mock_local.get_model.assert_called_with(model_name, version)
        mock_remote.get_model.assert_not_called()
        mock_local.save_model.assert_not_called()

    def test_get_model_from_remote(self):
        """Test getting a model when it only exists in remote repository."""
        # Arrange
        mock_local = MagicMock(spec=LocalCacheRepository)
        mock_remote = MagicMock(spec=S3Repository)

        mock_model = MagicMock(spec=tf.keras.Model)
        mock_local.get_model.side_effect = FileNotFoundError("Not in local")
        mock_remote.get_model.return_value = mock_model

        repo = CachingModelRepository(
            local_repository=mock_local,
            remote_repository=mock_remote
        )
        model_name = "test_model"
        version = "v1"

        # Act
        model = repo.get_model(model_name, version)

        # Assert
        assert model == mock_model
        mock_local.get_model.assert_called_with(model_name, version)
        mock_remote.get_model.assert_called_with(model_name, version)
        mock_local.save_model.assert_called_with(
            model_name, version, mock_model)

    def test_get_model_cache_failure(self):
        """Test handling when caching fails but model is still returned."""
        # Arrange
        mock_local = MagicMock(spec=LocalCacheRepository)
        mock_remote = MagicMock(spec=S3Repository)

        mock_model = MagicMock(spec=tf.keras.Model)
        mock_local.get_model.side_effect = FileNotFoundError("Not in local")
        mock_remote.get_model.return_value = mock_model
        mock_local.save_model.side_effect = Exception("Failed to save")

        repo = CachingModelRepository(
            local_repository=mock_local,
            remote_repository=mock_remote
        )
        model_name = "test_model"
        version = "v1"

        # Act
        model = repo.get_model(model_name, version)

        # Assert
        assert model == mock_model
        mock_local.get_model.assert_called_with(model_name, version)
        mock_remote.get_model.assert_called_with(model_name, version)
        mock_local.save_model.assert_called_with(
            model_name, version, mock_model)

    def test_get_model_latest_version_resolution(self):
        """Test resolving the latest version when requesting 'latest'."""
        # Arrange
        mock_local = MagicMock(spec=LocalCacheRepository)
        mock_remote = MagicMock(spec=S3Repository)

        mock_model = MagicMock(spec=tf.keras.Model)
        mock_local.get_model.return_value = mock_model

        repo = CachingModelRepository(
            local_repository=mock_local,
            remote_repository=mock_remote
        )
        model_name = "test_model"

        # Mock get_available_versions
        with patch.object(repo, 'get_available_versions', return_value=["v1", "v2", "v3"]) as mock_get_versions:
            # Act
            model = repo.get_model(model_name, "latest")

            # Assert
            assert model == mock_model
            mock_get_versions.assert_called_with(model_name)
            mock_local.get_model.assert_called_with(model_name, "v3")

    def test_get_model_no_versions(self):
        """Test error when no versions are available."""
        # Arrange
        mock_local = MagicMock(spec=LocalCacheRepository)
        mock_remote = MagicMock(spec=S3Repository)

        repo = CachingModelRepository(
            local_repository=mock_local,
            remote_repository=mock_remote
        )
        model_name = "test_model"

        # Mock get_available_versions
        with patch.object(repo, 'get_available_versions', return_value=[]) as mock_get_versions:
            # Act & Assert
            with pytest.raises(FileNotFoundError, match=f"No versions found for model '{model_name}'"):
                repo.get_model(model_name, "latest")
