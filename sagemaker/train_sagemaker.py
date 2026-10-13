"""
AWS SageMaker training entry point for RetainIQ.

Usage (from local machine with AWS CLI configured):

    # 1. Package source
    tar -czf sourcedir.tar.gz -C . app ml requirements.txt

    # 2. Upload data to S3
    aws s3 cp data/telecom_customers.csv s3://YOUR-BUCKET/retainiq/input/

    # 3. Run training job
    python sagemaker/train_sagemaker.py \
        --bucket YOUR-BUCKET \
        --role arn:aws:iam::ACCOUNT:role/SageMakerRole

This script uses the SageMaker SDK to launch a managed training job
on a ml.m5.large instance. The trained model artifact is uploaded to S3.
"""

import argparse
import os
import sagemaker
from sagemaker.sklearn.estimator import SKLearn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--instance", default="ml.m5.large")
    args = parser.parse_args()

    session = sagemaker.Session()
    role = args.role

    estimator = SKLearn(
        entry_point="train.py",
        source_dir="ml",
        role=role,
        instance_count=1,
        instance_type=args.instance,
        framework_version="1.2-1",
        py_version="py3",
        hyperparameters={"test_size": 0.2, "seed": 42},
        output_path=f"s3://{args.bucket}/retainiq/output/",
        base_job_name="retainiq-churn",
    )

    estimator.fit({
        "train": f"s3://{args.bucket}/retainiq/input/telecom_customers.csv"
    })

    print("Training job complete.")
    print(f"Model artifact: {estimator.model_data}")


if __name__ == "__main__":
    main()