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
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app


# Install only essential system dependencies first to improve layer caching
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry and configure it - separate layer for better caching
RUN curl -sSL https://install.python-poetry.org | python3 - 
ENV PATH="${POETRY_HOME}/bin:${PATH}"

RUN poetry --version \
    && poetry config virtualenvs.create false \ 
    && poetry config installer.parallel true \
    && poetry config installer.max-workers 10


# Copy only the dependency files first to leverage Docker cache
COPY pyproject.toml poetry.lock* README.md /app/

# Create directory for models
RUN mkdir -p /app/models/local \
    && mkdir -p /app/models/s3

# -----------------------------
# Base dependencies stage - only production deps 
# -----------------------------
FROM setup AS dependencies

# Install system dependencies required for TensorFlow and OpenCV
# Only install these in stages that need them
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install production dependencies
RUN poetry install --no-interaction --no-ansi --without dev --no-root

# -----------------------------
# Dev dependencies stage - adds dev dependencies 
# -----------------------------
FROM dependencies AS dev-dependencies

# Install dev dependencies using cache mount to speed up installation
RUN --mount=type=cache,target=/tmp/poetry_cache \
    poetry install --no-interaction --no-ansi --no-root

# -----------------------------
# Development stage - includes dev dependencies
# -----------------------------
FROM dev-dependencies AS development

# Copy the application code
COPY app /app/app

# Default command for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# -----------------------------
# Test stage - configured for running tests
# -----------------------------
FROM development AS test

# Set test environment variables
ENV PYTHONPATH=/app \
    TESTING=True


# Copy tests directory - separate layer from dev stage for better caching
COPY tests /app/tests

# Default command for running all tests
CMD ["python", "-m", "pytest"]

# -----------------------------
# Unit test stage - only runs unit tests (same image as test, different command)
# -
FROM test AS unit-test

# Command for running only unit tests
CMD ["python", "-m", "pytest", "-k", "not integration", "-v"]

# -----------------------------
# Integration test stage - only runs integration tests (same image as test, different command)
# -----------------------------
FROM test AS integration-test

# Command for running only integration tests
CMD ["python", "-m", "pytest", "-m", "integration", "-v"]

# -----------------------------
# Production stage - minimal dependencies
# -----------------------------
FROM dependencies AS production

# Copy application code (excluding tests and dev files)
COPY app /app/app
COPY scripts /app/scripts

# Make entrypoint script executable
RUN chmod +x /app/scripts/entrypoint.sh

# Install AWS CLI for S3 access
RUN pip install boto3

# Expose the port
EXPOSE 8000

# Set the entrypoint to download models before starting the app
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# Default command (passed to entrypoint script)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
