import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import numpy as np

from app.services.general_classifier import GeneralClassifierService
from app.utils.inference_models.model_repository import LocalCacheRepository, CachingModelRepository, ModelRepository
from app.utils.inference_models.repository_factory import create_model_repository


class TestGeneralClassifierService(unittest.TestCase):
    """Test cases for the GeneralClassifierService"""

    def setUp(self):
        """Set up for each test case"""
        # Load the sample test image
        test_image_path = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'test_data', 'test_dog.jpg')
        with open(test_image_path, 'rb') as f:
            self.sample_image_data = f.read()

    @patch('app.services.general_classifier.config')
    def test_initialization(self, mock_config_general_classifier: MagicMock):
        """Test that the service initializes correctly"""
        # Configure mocks
        mock_config_general_classifier.CAT_DOG_OTHER_CLASSIFIER = "test_model"

        # Create a mock repository
        mock_model = MagicMock()
        mock_repo = MagicMock(spec=ModelRepository)
        mock_repo.get_model.return_value = mock_model

        # Initialize the service with our mock repository
        service = GeneralClassifierService(mock_repo)

        # Verify initialization
        self.assertEqual(service.model_name, "test_model")
        self.assertEqual(service.model, mock_model)
        mock_repo.get_model.assert_called_once_with("test_model")

    @patch('app.services.general_classifier.config')
    def test_predict_success(self, mock_config: MagicMock):
        """Test successful prediction"""
        # Configure mocks
        mock_config.CAT_DOG_OTHER_CLASSIFIER = "test_model"

        # Create a mock repository
        mock_model = MagicMock()
        mock_repo = MagicMock(spec=ModelRepository)
        mock_repo.get_model.return_value = mock_model

        # Initialize the service with our mock repository
        service = GeneralClassifierService(model_repository=mock_repo)

        # Configure mock model to return fake prediction results
        # Dog: 70%, Cat: 20%, Other: 10%
        mock_prediction = np.array([[0.7, 0.2, 0.1]])
        mock_model.predict.return_value = mock_prediction

        # Call predict and verify results
        result = service.predict(self.sample_image_data)

        # Verify the result format
        self.assertIsInstance(result, dict)
        self.assertTrue(all(isinstance(k, str) for k in result.keys()))
        self.assertTrue(all(isinstance(v, float) for v in result.values()))

        # Check that we have the expected classes
        expected_classes = ["dog", "cat", "other"]
        for cls in expected_classes:
            self.assertIn(cls, result)

        # Verify probabilities sum to approximately 1.0
        self.assertAlmostEqual(sum(result.values()), 1.0, places=7)

    @patch('app.services.general_classifier.config')
    def test_with_local_repository(self, mock_config: MagicMock):
        """Test that the service works with a local repository"""
        # Configure mocks
        mock_config.CAT_DOG_OTHER_CLASSIFIER = "test_model"

        # Create a mock local repository
        mock_model = MagicMock()
        # Using LocalCacheRepository spec for more type safety
        mock_repo = MagicMock(spec=LocalCacheRepository)
        mock_repo.get_model.return_value = mock_model

        # Initialize the service with our mock repository
        service = GeneralClassifierService(model_repository=mock_repo)

        # Configure mock model to return fake prediction results
        mock_prediction = np.array(
            [[0.1, 0.8, 0.1]])  # Cat: 10%, Dog: 80%, Other: 10%
        mock_model.predict.return_value = mock_prediction

        # Call predict and verify results
        result = service.predict(self.sample_image_data)

        # Check prediction values
        self.assertAlmostEqual(result['cat'], 0.1)
        self.assertAlmostEqual(result['dog'], 0.8)
        self.assertAlmostEqual(result['other'], 0.1)

    @patch('app.utils.inference_models.repository_factory.os')
    def test_repository_factory(self, mock_os):
        """Test that the repository factory creates the correct repository type"""
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with USE_LOCAL_MODEL_REPO=true
            mock_os.getenv.side_effect = lambda key, default=None: {
                'USE_LOCAL_MODEL_REPO': 'true',
                'TESTING': 'false',
                'MODEL_REPOSITORY_TYPE': 'caching',
                'TEST_MODELS_DIR': temp_dir,
                'MODELS_DIR': temp_dir
            }.get(key, default)

            # We need to mock the directory creation logic in the repository to avoid filesystem access
            with patch('os.path.exists', return_value=True):
                repo = create_model_repository()
                self.assertIsInstance(repo, LocalCacheRepository)

            # Test with TESTING=true
            mock_os.getenv.side_effect = lambda key, default=None: {
                'USE_LOCAL_MODEL_REPO': 'false',
                'TESTING': 'true',
                'MODEL_REPOSITORY_TYPE': 'caching',
                'TEST_MODELS_DIR': temp_dir,
                'MODELS_DIR': temp_dir
            }.get(key, default)

            # We need to mock the directory creation logic in the repository to avoid filesystem access
            with patch('os.path.exists', return_value=True):
                repo = create_model_repository()
                self.assertIsInstance(repo, LocalCacheRepository)

            # Test with normal configuration
            mock_os.getenv.side_effect = lambda key, default=None: {
                'USE_LOCAL_MODEL_REPO': 'false',
                'TESTING': 'false',
                'MODEL_REPOSITORY_TYPE': 'caching',
                'TEST_MODELS_DIR': temp_dir,
                'MODELS_DIR': temp_dir
            }.get(key, default)

            # This test would need proper mocking of CachingModelRepository's dependencies
            # as it will try to initialize both local and S3 repositories
            with patch('app.utils.inference_models.repository_factory.CachingModelRepository') as mock_cache_repo, \
                    patch('os.path.exists', return_value=True):
                mock_cache_repo.return_value = MagicMock()
                create_model_repository()
                mock_cache_repo.assert_called_once()


if __name__ == '__main__':
    unittest.main()
