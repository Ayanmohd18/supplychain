#!/bin/bash
# Sets up EC2 t2.micro with Docker, pulls model from S3, starts API

set -e

echo "Updating system packages..."
sudo yum update -y

echo "Installing Docker..."
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user
sudo chkconfig docker on

echo "Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Environment variables (to be injected or set via SSM Parameter Store)
S3_BUCKET="s3://supplychain-m5-models/models/tft/latest/"
LOCAL_MODEL_DIR="/opt/models"
LOCAL_DATA_DIR="/opt/data"

echo "Creating local directories..."
sudo mkdir -p $LOCAL_MODEL_DIR
sudo mkdir -p $LOCAL_DATA_DIR
sudo chown -R ec2-user:ec2-user $LOCAL_MODEL_DIR $LOCAL_DATA_DIR

echo "Pulling model artifact from S3..."
aws s3 sync $S3_BUCKET $LOCAL_MODEL_DIR

echo "Starting forecast API container..."
# Assuming the image is available locally or pulled from ECR
docker run -d \
    --name forecast-api \
    -p 8000:8000 \
    -e MODEL_DIR=$LOCAL_MODEL_DIR \
    -e DATA_DIR=$LOCAL_DATA_DIR \
    -v $LOCAL_MODEL_DIR:$LOCAL_MODEL_DIR \
    -v $LOCAL_DATA_DIR:$LOCAL_DATA_DIR \
    forecast-api:latest

echo "Setup complete. API is running on port 8000."
