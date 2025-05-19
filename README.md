[![CI Pipeline](https://github.com/aarangop/wmgb-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/aarangop/wmgb-backend/actions/workflows/ci.yml)

# Who's My Good Boy - AI Image Classification Service

A FastAPI-based backend service for classifying images using TensorFlow models,
primarily focused on animal classification.

## Features

- Cat-Dog-Other classification with fine-tuned TensorFlow models
- "Apolo" (specific dog) detection endpoint
- Clean API with proper error handling
- Comprehensive test suite (unit and integration tests)
- Poetry for dependency management
- Multi-stage Docker build for development, testing, and production
- AWS S3 integration for model storage
- CI/CD with GitHub Actions

## Setup

### Prerequisites

- Python 3.10 or later (but less than 3.12)
- [Poetry](https://python-poetry.org/) for dependency management
- Docker and Docker Compose (optional, for containerized development)

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd wmgb-backend
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

3. Activate the Poetry virtual environment:
   ```bash
   poetry shell
   ```

### Environment Configuration

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your configuration
   - For Docker development, also create a `.docker.env` file

### Running the Application

#### Using Poetry

```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Using Docker Compose

```bash
docker-compose up dev
```

The API will be available at http://localhost:8000

#### API Documentation

Once the application is running, you can access:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Running Tests

#### Using Poetry

```bash
# Run all tests
poetry run pytest

# Run unit tests only
poetry run pytest -k "not integration"

# Run integration tests only
poetry run pytest -m integration

# Run tests with coverage
poetry run pytest --cov=app --cov-report=term --cov-report=html
```

#### Using VS Code Tasks

The project includes predefined VS Code tasks:

- Run Unit Tests
- Run Integration Tests
- Run All Tests
- Run Tests with Coverage

#### Using Docker Compose

```bash
# Run unit tests
docker-compose run unit-tests

# Run integration tests
docker-compose run integration-tests
```

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /api/v1/{version}/classify` - General image classification
- `POST /api/v1/{version}/is-apolo` - Apolo-specific dog detection

## Docker

The project uses a multi-stage Dockerfile for different environments:

- **Development**: Includes all dependencies and hot-reload
- **Test**: Configured for running tests
- **Unit Test**: Runs only unit tests
- **Integration Test**: Runs only integration tests
- **Production**: Minimal dependencies for production deployment

### Building and Running with Docker

```bash
# Build and run the development environment
docker-compose up dev

# Build and run the production environment
docker build -t wmgb-backend --target production .
docker run -p 8000:8000 -e PORT=8000 wmgb-backend
```

## CI/CD with GitHub Actions

This project uses GitHub Actions for CI/CD. The workflow is defined in
`.github/workflows/ci.yml`.

### Environment Variables for CI/CD

For integration tests to work in GitHub Actions, set up the following repository
secrets:

1. Go to your GitHub repository settings
2. Click on "Secrets and variables" -> "Actions"
3. Add the following secrets:
   - `AWS_REGION`: The AWS region (e.g., "us-east-2")
   - `S3_MODELS_BUCKET`: The S3 bucket name for models
   - `AWS_ACCESS_KEY_ID`: Your AWS access key
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret key

The workflow runs unit tests and integration tests separately. Unit tests will
run first, and integration tests will only run if the unit tests pass.

### Testing GitHub Actions Locally

You can test GitHub Actions locally using [act](https://github.com/nektos/act):

1. Create a `.ci.env` file with your CI environment variables
2. Run the provided script:

```bash
./scripts/run-github-actions-locally.sh
```

To run specific jobs:

```bash
# Run only unit tests
./scripts/run-github-actions-locally.sh -j unit-tests

# Run only integration tests
./scripts/run-github-actions-locally.sh -j integration-tests
```

### Utility Scripts

The project includes several utility scripts in the `scripts/` directory:

- `check-s3.sh` - Verify S3 connection and bucket access
- `debug-docker.sh` - Debug Docker container issues
- `docker-compose-test.sh` - Run tests in Docker Compose
- `fast-test.sh` - Run a quick test suite
- `rebuild.sh` - Rebuild Docker containers
- `run-docker.sh` - Run the Docker container
- `run-github-actions-locally.sh` - Test GitHub Actions locally
- `setup-github-actions.sh` - Set up GitHub Actions
- `update-and-build.sh` - Update dependencies and build

### Environment Variables for Development

For local development:

1. Copy `.env.example` to `.env`
2. Fill in your values for local development

For Docker:

1. Copy `.env.example` to `.docker.env`
2. Fill in your values for Docker environment
