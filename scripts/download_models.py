#!/usr/bin/env python
"""
Script to download model files from S3 at container startup.
This ensures the latest models are always used in production.
"""

import os
import logging
import boto3
import botocore
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_downloader")


def download_models_from_s3():
    """
    Downloads model files from S3 bucket to the local models directory.
    Reads configuration from environment variables.
    """
    # Get configuration from environment variables
    s3_bucket = os.environ.get("MODEL_S3_BUCKET")
    s3_prefix = os.environ.get("MODEL_S3_PREFIX", "models/")
    models_dir = os.environ.get("MODELS_DIR", "/app/models")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")

    if not s3_bucket:
        logger.warning(
            "MODEL_S3_BUCKET environment variable not set. Skipping model download.")
        return False

    logger.info(
        f"Starting model download from s3://{s3_bucket}/{s3_prefix} to {models_dir}")

    # Ensure models directory exists
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Create S3 client
        s3_client = boto3.client('s3', region_name=aws_region)

        # List objects in bucket with prefix
        response = s3_client.list_objects_v2(
            Bucket=s3_bucket, Prefix=s3_prefix)

        if 'Contents' not in response:
            logger.warning(f"No models found in s3://{s3_bucket}/{s3_prefix}")
            return False

        # Download each file
        for obj in response['Contents']:
            # Skip directories (objects that end with '/')
            if obj['Key'].endswith('/'):
                continue

            # Get relative path (remove prefix)
            rel_path = obj['Key'][len(s3_prefix):] if obj['Key'].startswith(
                s3_prefix) else obj['Key']
            local_file_path = os.path.join(models_dir, rel_path)

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            logger.info(f"Downloading {obj['Key']} to {local_file_path}")

            # Download file with progress callback
            s3_client.download_file(
                s3_bucket,
                obj['Key'],
                local_file_path
            )

            logger.info(f"Successfully downloaded {obj['Key']}")

        logger.info("All models downloaded successfully")
        return True

    except botocore.exceptions.ClientError as e:
        logger.error(f"S3 download error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading models: {e}")
        return False


if __name__ == "__main__":
    # Try to download models
    success = download_models_from_s3()

    # If models downloaded successfully, continue with execution
    if success:
        logger.info("Models downloaded successfully, starting application...")
    else:
        logger.warning("Continuing without downloading new models...")

    # Execute the passed command (typically starting the application)
    import sys
    if len(sys.argv) > 1:
        import subprocess
        cmd = sys.argv[1:]
        logger.info(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd)
