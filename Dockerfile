FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for TensorFlow and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy configuration files first for better caching
COPY pyproject.toml poetry.lock README.md /app/

# Configure poetry
RUN poetry config virtualenvs.create false
RUN poetry config installer.parallel true
RUN poetry config installer.max-workers 10

# Copy application code
COPY app /app/app
# Copy test directory for CI
COPY tests /app/tests
# Copy any model files if they exist
COPY models/ /app/models/

# Install all dependencies including dev for testing (this will use the existing lock file)
RUN poetry install --no-interaction --no-ansi

# Set environment variables
# These will be overridden by values passed at runtime
ENV S3_MODELS_BUCKET=""
ENV AWS_REGION=""
ENV MODEL_SOURCE=""
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV PORT=8000

# Expose the application port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]