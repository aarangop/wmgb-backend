#!/bin/bash

# Script to check S3 for the existence of the model
echo "Checking S3 for model existence..."

# Navigate to the script directory
cd "$(dirname "$0")"

# Load environment variables from .docker.env
export $(grep -v '^#' .docker.env | xargs)

# List objects in the S3 bucket
echo "Listing objects in bucket ${S3_MODELS_BUCKET}:"
aws s3 ls s3://${S3_MODELS_BUCKET}/ --recursive | grep -i classifier

# Print environment variables that are used for model loading
echo -e "\nEnvironment variables:"
echo "MODEL_SOURCE: ${MODEL_SOURCE}"
echo "S3_MODELS_BUCKET: ${S3_MODELS_BUCKET}"
echo "CAT_DOG_OTHER_CLASSIFIER: ${CAT_DOG_OTHER_CLASSIFIER}"
echo "AWS_REGION: ${AWS_REGION}"
