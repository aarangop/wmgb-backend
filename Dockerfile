# Who's My Good Boy" project

# -----------------------------
# Setup stage - Download and install dependencies
# -----------------------------
FROM python:3.10-slim AS setup

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_NO_INTERACTION=1

# Install system dependencies required for TensorFlow and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Configure poetry
RUN poetry config virtualenvs.create false
RUN poetry config installer.parallel true
RUN poetry config installer.max-workers 10

# Set working directory
WORKDIR /app

COPY app /app/app

# -----------------------------
# Base dependencies stage - only production deps 
# -----------------------------
FROM setup as dependencies

# Copy pyproject.toml and files required by poetry
COPY pyproject.toml poetry.lock* README.md /app/

RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi --without dev --no-root

# -----------------------------
# Dev dependencies stage - adds dev dependencies 
# -----------------------------
FROM dependencies as dev-dependencies

RUN poetry install --no-interaction --no-ansi --no-root

# -----------------------------
# Development stage - includes dev dependencies
# -----------------------------
FROM dev-dependencies AS development

# Install all dependencies including development
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi

# Copy the rest of the application
COPY app /app/app
COPY models/ /app/models

# Default command for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# # -----------------------------
# # Test stage - configured for running tests
# # -----------------------------
FROM development AS test

# Set test environment variables
ENV PYTHONPATH=/app \
    TESTING=True

COPY tests /app/tests

# Default command for running all tests
CMD ["python", "-m", "pytest"]

# -----------------------------
# Unit test stage - only runs unit tests
# -----------------------------
FROM test AS unit-test

# Command for running only unit tests
CMD ["python", "-m", "pytest", "-k", '"not integration"', "-v"]

# -----------------------------
# Integration test stage - only runs integration tests
# -----------------------------
FROM test AS integration-test

# Command for running only integration tests
CMD ["python", "-m", "pytest", "-m", "integration", "-v"]

# -----------------------------
# Production stage - minimal dependencies
# -----------------------------
FROM dependencies AS production

# Create directory for models
RUN mkdir -p /app/models

# Copy application code (excluding tests and dev files)
COPY app /app/app
COPY scripts /app/scripts

# Copy the entrypoint script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Set environment variables for production
ENV MODEL_PATH="/app/models" \
    S3_BUCKET_NAME="whos-my-good-boy-models" \
    MODEL_LOADING_STRATEGY="s3"

# Expose the port
EXPOSE 8000

# Use entrypoint script to handle model loading and startup
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
