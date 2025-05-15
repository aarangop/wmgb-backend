# Who's My Good Boy - AI Image Classification Service

A FastAPI-based backend service for classifying images using various AI models.

## Features

- General image classification endpoint
- Dog breed detection endpoint
- "Apolo" (specific dog) detection endpoint
- Clean API with proper error handling
- Comprehensive test suite

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py      # Health check endpoint
│   │   │   └── predictions.py # Classification endpoints
│   │   └── dependencies.py    # Dependency injection
│   ├── core/
│   │   ├── config.py          # App configuration
│   │   └── errors.py          # Error handling
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   ├── services/
│   │   ├── base.py            # Base classifier service
│   │   ├── general_classifier.py # General image classifier
│   │   ├── dog_classifier.py     # Dog detection
│   │   └── apolo_classifier.py   # Apolo detection
│   └── main.py                # FastAPI app
├── tests/
│   ├── api/
│   │   ├── test_health.py
│   │   └── test_predictions.py
│   └── test_data/             # Test images
└── requirements.txt           # Dependencies
```

## Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd whos-my-good-boy/backend
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the application

```
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

### Running tests

```
pytest
```

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /api/v1/classify` - General image classification
- `POST /api/v1/is-dog` - Dog detection
- `POST /api/v1/is-apolo` - Apolo detection

## Docker

You can build and run the application using Docker:

```
docker build -t whos-my-good-boy-api .
docker run -p 8000:8000 whos-my-good-boy-api
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

This will run the GitHub Actions workflow locally using the environment
variables from `.ci.env`.

To run specific jobs:

```bash
# Run only unit tests
./scripts/run-github-actions-locally.sh -j unit-tests

# Run only integration tests
./scripts/run-github-actions-locally.sh -j integration-tests
```

### Environment Variables for Development

For local development:

1. Copy `.env.example` to `.env`
2. Fill in your values for local development

For Docker:

1. Copy `.env.example` to `.docker.env`
2. Fill in your values for Docker environment
