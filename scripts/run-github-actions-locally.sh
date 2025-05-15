#!/bin/bash

# This script runs GitHub Actions locally using act and loads environment variables from .ci.env

# Navigate to the script directory
cd "$(dirname "$0")/.."

# Check if act is installed
if ! command -v act &> /dev/null; then
    echo "Error: 'act' is not installed. Please install it to run GitHub Actions locally."
    echo "For macOS: brew install act"
    echo "For other platforms: https://github.com/nektos/act#installation"
    exit 1
fi

# Check if .ci.env exists
if [[ ! -f .ci.env ]]; then
    echo "Error: .ci.env file not found. Please create one with your CI environment variables."
    exit 1
fi

echo "Running GitHub Actions locally using act and environment variables from .ci.env..."

# Run act with the .ci.env file
# The --env-file flag tells act to load environment variables from .ci.env
act --env-file .ci.env "$@"

# Note: You can specify which job to run by adding it as an argument:
# Example: ./scripts/run-github-actions-locally.sh -j unit-tests
# Example: ./scripts/run-github-actions-locally.sh -j integration-tests
