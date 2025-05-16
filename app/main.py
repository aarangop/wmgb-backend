import time
import uuid
from loguru import logger
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, predictions
from app.core.config import config
from app.core.logging import setup_logging

logger.info("Starting FastAPI application...")
logger.info(f"Environment: {config.ENV}")
logger.info(f"API Version: {config.API_VERSION}")
logger.info(f"Model Source: {config.USE_LOCAL_MODEL_REPO}")
logger.info(f"Models Directory: {config.MODELS_DIR}")
logger.info(f"Cat-dog-other classifier: {config.CAT_DOG_OTHER_CLASSIFIER}")
logger.info(f"Log Level: {config.LOG_LEVEL}")

app = FastAPI(
    title="Who's My Good Boy API",
    description="AI service for image classification",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging with settings
setup_logging(log_level=config.LOG_LEVEL, json_logs=config.JSON_LOGS)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware to log all requests and responses"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Log request details
    logger.info(f"Request {request_id} - {request.method} {request.url.path}")

    # Measure time
    start_time = time.time()

    # Process request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log successful response
        logger.info(
            f"Response {request_id} - Status: {response.status_code} - Took: {process_time:.3f}s")

        return response
    except Exception as e:
        # Log exceptions
        process_time = time.time() - start_time
        logger.error(
            f"Response {request_id} - Exception: {str(e)} - Took: {process_time:.3f}s")
        raise


# Include routers
app.include_router(health.router)
app.include_router(predictions.router,
                   prefix=f"/api/{config.API_VERSION}",
                   tags=["predictions"])

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=config.PORT, reload=True)

# To run this application from the command line:
# If you are in the backend directory:
# uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
#
# If you are in the project root directory:
# uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
