import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np

from app.services.general_classifier import GeneralClassifierService
from app.utils.inference_models.model_repository import LocalCacheRepository, CachingModelRepository


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
    @patch('app.services.general_classifier.CachingModelRepository')
    def test_initialization(
            self,
            mock_caching_repository: MagicMock,
            mock_config_general_classifier: MagicMock):
        """Test that the service initializes correctly"""
        # Configure mocks
        mock_config_general_classifier.CAT_DOG_OTHER_CLASSIFIER = "test_model"
        mock_model = MagicMock()
        mock_repo_instance = MagicMock()
        mock_caching_repository.return_value = mock_repo_instance
        mock_repo_instance.get_model.return_value = mock_model

        # Initialize the service with mocks in place
        service = GeneralClassifierService()

        # Verify initialization
        self.assertEqual(service.model_name, "test_model")
        self.assertEqual(service.model, mock_model)
        mock_repo_instance.get_model.assert_called_once_with("test_model")

    @patch('app.services.general_classifier.config')
    @patch('app.services.general_classifier.CachingModelRepository')
    def test_predict_success(self, mock_caching_repository: MagicMock, mock_config: MagicMock):
        """Test successful prediction"""
        # Configure mocks
        mock_config.CAT_DOG_OTHER_CLASSIFIER = "test_model"
        mock_model = MagicMock()
        mock_repo_instance = MagicMock()
        mock_caching_repository.return_value = mock_repo_instance
        mock_repo_instance.get_model.return_value = mock_model

        # Initialize the service with mocks in place
        service = GeneralClassifierService()

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
    @patch('app.services.general_classifier.CachingModelRepository')
    @patch('app.utils.inference_models.model_repository.LocalCacheRepository')
    def test_with_local_repository(self, mock_local_repository: MagicMock,
                                   mock_caching_repository: MagicMock,
                                   mock_config: MagicMock):
        """Test that the service works with a local repository"""
        # Configure mocks
        mock_config.CAT_DOG_OTHER_CLASSIFIER = "test_model"

        # Set up the local repository mock
        mock_model = MagicMock()
        mock_local_repo_instance = MagicMock()
        mock_local_repository.return_value = mock_local_repo_instance
        mock_local_repo_instance.get_model.return_value = mock_model

        # Make CachingModelRepository use our mocked local repository
        mock_repo_instance = MagicMock()
        mock_caching_repository.return_value = mock_repo_instance
        mock_repo_instance.get_model.return_value = mock_model

        # Initialize the service with mocks in place
        service = GeneralClassifierService()

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


if __name__ == '__main__':
    unittest.main()
