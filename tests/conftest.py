import pytest
import sys
from loguru import logger
from fastapi.testclient import TestClient
from app.main import app

# Import for ModelLoaderManager reset
try:
    from app.utils.inference_models.model_loader_manager import ModelLoaderManager
except ImportError:
    # For tests that don't need ModelLoaderManager
    ModelLoaderManager = None


@pytest.fixture(scope="session", autouse=True)
def setup_logger():
    """
    Set up the logger globally before all tests run.
    This fixture runs automatically before any tests.
    """
    # Remove any existing handlers
    logger.remove()

    # Add a handler that writes to stderr with a specific format
    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                "level": "INFO",
            }
        ]
    )

    # You can also create a test-specific log file if needed
    # logger.add("tests/logs/test.log", rotation="10 MB", level="DEBUG")

    logger.info("Logger has been configured for tests")

    yield

    # Cleanup if necessary
    logger.info("Test session completed")


@pytest.fixture
def client():
    """
    Create a test client for the FastAPI application
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def reset_model_loader_manager():
    """
    Reset the ModelLoaderManager singleton before each test that imports it.
    This helps prevent test pollution from one test to another.
    """
    if ModelLoaderManager is not None:
        # Reset the singleton instance
        ModelLoaderManager._loader = None

    yield

    # Also reset after test
    if ModelLoaderManager is not None:
        ModelLoaderManager._loader = None
