import logging
import sys
from pathlib import Path
from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentation.
    See https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(log_level="INFO", json_logs=False):
    """Configure logging with loguru"""
    # Remove all existing handlers
    logging.root.handlers = []

    # Set log level
    logging.root.setLevel(log_level)

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Configure loguru
    log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"

    # JSON format if needed for production environments
    if json_logs:
        log_format = "{time} | {level} | {message}"
        logger.configure(
            handlers=[
                {"sink": sys.stdout, "serialize": True},
                {"sink": "logs/app.log", "serialize": True,
                    "rotation": "2 MB", "retention": "1 week"},
            ]
        )
    else:
        logger.configure(
            handlers=[
                {"sink": sys.stdout, "format": log_format},
                {"sink": "logs/app.log", "format": log_format,
                    "rotation": "10 MB", "retention": "1 week"},
            ]
        )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Replace standard library logging
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
