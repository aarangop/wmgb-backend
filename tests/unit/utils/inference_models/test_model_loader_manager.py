import os
import pytest
from unittest.mock import patch, MagicMock

from app.utils.inference_models.model_loader_manager import ModelLoaderManager


class TestModelLoaderManager:

    def test_get_loader_development_environment(self):
        """Test getting loader for local model"""
        # Reset the singleton instance before applying any patches
        ModelLoaderManager._loader = None

        with patch("app.utils.inference_models.model_loader_manager.config") as mock_config:
            mock_config.MODEL_SOURCE = "local"
            mock_config.MODELS_DIR = "tests/test_data/models"

            with patch('app.utils.inference_models.model_loader_manager.LocalModelLoader') as mock_loader:
                # Create a mock instance
                mock_instance = MagicMock()
                mock_loader.return_value = mock_instance

                # Act
                loader = ModelLoaderManager.get_loader()

                # Assert
                mock_loader.assert_called_once_with("tests/test_data/models")
                assert loader == mock_instance

    def test_get_loader_from_s3(self):
        """Test that getting an s3 model loader"""
        # Reset the singleton instance before applying any patches
        ModelLoaderManager._loader = None

        with patch("app.utils.inference_models.model_loader_manager.config") as mock_config:
            mock_config.MODEL_SOURCE = "s3"

            with patch("app.utils.inference_models.model_loader_manager.S3ModelLoader") as mock_s3_model_loader:
                # Create a mock instance
                mock_instance = MagicMock()
                mock_s3_model_loader.return_value = mock_instance

                # Act
                loader = ModelLoaderManager.get_loader()

                # Assert
                mock_s3_model_loader.assert_called_once()
                assert loader == mock_instance

    def test_get_loader_singleton_pattern(self):
        """Test that get_loader returns the same instance when called multiple times"""
        # Reset the singleton instance before applying any patches
        ModelLoaderManager._loader = None

        with patch("app.utils.inference_models.model_loader_manager.config") as mock_config:
            # Arrange
            mock_config.MODEL_SOURCE = 'local'

            with patch('app.utils.inference_models.model_loader_manager.LocalModelLoader') as mock_loader:
                # Create a mock instance
                mock_instance = MagicMock()
                mock_loader.return_value = mock_instance

                # Act
                first_call = ModelLoaderManager.get_loader()
                second_call = ModelLoaderManager.get_loader()

                # Assert
                mock_loader.assert_called_once()  # Should only be instantiated once
                assert first_call == second_call

    def test_get_loader_default_environment(self):
        """test getting loader when environment is not set"""
        # Reset the singleton instance before applying any patches
        ModelLoaderManager._loader = None

        # Use context managers for more controlled patching
        with patch("app.utils.inference_models.model_loader_manager.config") as mock_config:
            # Configure mock_config with concrete values
            mock_config.MODEL_SOURCE = "local"
            mock_config.MODELS_DIR = "tests/test_data/models"

            with patch('app.utils.inference_models.model_loader_manager.LocalModelLoader') as mock_loader:
                # Create a mock instance
                mock_instance = MagicMock()
                mock_loader.return_value = mock_instance

                # Act
                loader = ModelLoaderManager.get_loader()

                # Assert
                mock_loader.assert_called_once_with("tests/test_data/models")
                assert loader == mock_instance
