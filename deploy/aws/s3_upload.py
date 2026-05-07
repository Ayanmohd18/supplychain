"""
Upload trained model and features to S3.
Versioned by timestamp: s3://bucket/models/tft/YYYYMMDD_HHMMSS/
"""

import os
import boto3
from datetime import datetime

def upload_to_s3(local_dir, bucket_name, s3_prefix):
    """Uploads a local directory to an S3 bucket."""
    s3_client = boto3.client('s3')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_prefix = f"{s3_prefix}/{timestamp}/"
    latest_prefix = f"{s3_prefix}/latest/"
    
    print(f"Uploading files from {local_dir} to s3://{bucket_name}/{versioned_prefix}")
    
    # Upload to versioned path and latest path
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            
            s3_versioned_key = os.path.join(versioned_prefix, relative_path).replace("\\", "/")
            s3_latest_key = os.path.join(latest_prefix, relative_path).replace("\\", "/")
            
            print(f"Uploading {local_path} -> {s3_versioned_key}")
            s3_client.upload_file(local_path, bucket_name, s3_versioned_key)
            
            print(f"Updating latest tag {local_path} -> {s3_latest_key}")
            s3_client.upload_file(local_path, bucket_name, s3_latest_key)

if __name__ == "__main__":
    # Configuration
    BUCKET_NAME = os.getenv("S3_BUCKET", "supplychain-m5-models")
    MODEL_DIR = os.getenv("MODEL_DIR", "../../lightning_logs/")
    FEATURES_DIR = os.getenv("FEATURES_DIR", "../../data/processed/")
    
    if not os.path.exists(MODEL_DIR):
        print(f"Warning: Model directory {MODEL_DIR} not found.")
    else:
        # Upload models
        upload_to_s3(MODEL_DIR, BUCKET_NAME, "models/tft")
        
    if not os.path.exists(FEATURES_DIR):
        print(f"Warning: Features directory {FEATURES_DIR} not found.")
    else:
        # Upload processed features
        upload_to_s3(FEATURES_DIR, BUCKET_NAME, "data/features")
    
    print("Upload completed successfully.")
