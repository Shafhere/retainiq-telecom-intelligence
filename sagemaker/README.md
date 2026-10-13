# AWS SageMaker Training

This folder contains the SageMaker entry point for RetainIQ.

## Prerequisites
- AWS account with SageMaker permissions
- S3 bucket
- SageMaker execution role
- Local: `pip install sagemaker boto3`

## Run

```bash
# 1. Package
tar -czf sourcedir.tar.gz -C . app ml requirements.txt

# 2. Upload data
aws s3 cp data/telecom_customers.csv s3://YOUR-BUCKET/retainiq/input/

# 3. Train
python sagemaker/train_sagemaker.py \
    --bucket YOUR-BUCKET \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole