import pytest
import sys
from loguru import logger
from fastapi.testclient import TestClient
from app.main import app


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
