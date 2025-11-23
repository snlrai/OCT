#!/usr/bin/env python3
"""
AWS Infrastructure Setup Script for OCT MLOps Pipeline

This script sets up the required AWS resources for the MLOps pipeline.
Run with --dry-run flag to preview changes without creating resources.

Usage:
    python scripts/setup_aws.py --dry-run  # Preview only
    python scripts/setup_aws.py           # Create resources
    python scripts/setup_aws.py --teardown # Delete all resources
"""

import os
import sys
import argparse
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from dotenv import load_dotenv
import json
from typing import Dict, List, Tuple

# Load environment variables
env_path = Path(__file__).parent.parent / 'aws_config' / '.env.aws'
if not env_path.exists():
    print(f"❌ Error: {env_path} not found!")
    print("Please copy .env.aws.template to .env.aws and fill in your AWS credentials.")
    sys.exit(1)

load_dotenv(env_path)

# Configuration
PROJECT_NAME = os.getenv('PROJECT_NAME', 'oct-mlops')
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
S3_DATA_BUCKET = os.getenv('S3_DATA_BUCKET')
S3_MODELS_BUCKET = os.getenv('S3_MODELS_BUCKET')
S3_ARTIFACTS_BUCKET = os.getenv('S3_ARTIFACTS_BUCKET')
DYNAMODB_EXPERIMENTS_TABLE = os.getenv('DYNAMODB_EXPERIMENTS_TABLE')
DYNAMODB_MODELS_TABLE = os.getenv('DYNAMODB_MODELS_TABLE')
DYNAMODB_TRAINING_RUNS_TABLE = os.getenv('DYNAMODB_TRAINING_RUNS_TABLE')

class AWSSetup:
    """Handles AWS resource creation and management"""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.s3_client = boto3.client('s3', region_name=AWS_REGION)
        self.dynamodb_client = boto3.client('dynamodb', region_name=AWS_REGION)
        self.iam_client = boto3.client('iam', region_name=AWS_REGION)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=AWS_REGION)
        
        self.created_resources = {
            's3_buckets': [],
            'dynamodb_tables': [],
            'iam_roles': []
        }
    
    def estimate_costs(self) -> Dict[str, str]:
        """Estimate monthly costs for the infrastructure"""
        return {
            'S3 Storage (5GB)': '$0.00 (Free Tier)',
            'DynamoDB (25GB + 25 RCU/WCU)': '$0.00 (Free Tier)',
            'Lambda (1M requests)': '$0.00 (Free Tier)',
            'EC2 t2.micro (750 hours)': '$0.00 (Free Tier)',
            'CloudWatch Logs (5GB)': '$0.00 (Free Tier)',
            'Data Transfer': '$0.00 - $1.00 (depends on usage)',
            'TOTAL ESTIMATED': '$0.00 - $1.00/month (within Free Tier)'
        }
    
    def create_s3_buckets(self) -> List[str]:
        """Create S3 buckets for data, models, and artifacts"""
        buckets = [S3_DATA_BUCKET, S3_MODELS_BUCKET, S3_ARTIFACTS_BUCKET]
        created = []
        
        for bucket_name in buckets:
            if self.dry_run:
                print(f"  [DRY RUN] Would create S3 bucket: {bucket_name}")
                continue
            
            try:
                # Check if bucket exists
                self.s3_client.head_bucket(Bucket=bucket_name)
                print(f"  ✓ S3 bucket already exists: {bucket_name}")
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    # Bucket doesn't exist, create it
                    try:
                        if AWS_REGION == 'us-east-1':
                            self.s3_client.create_bucket(Bucket=bucket_name)
                        else:
                            self.s3_client.create_bucket(
                                Bucket=bucket_name,
                                CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
                            )
                        
                        # Enable versioning for models bucket
                        if 'models' in bucket_name:
                            self.s3_client.put_bucket_versioning(
                                Bucket=bucket_name,
                                VersioningConfiguration={'Status': 'Enabled'}
                            )
                        
                        # Add lifecycle policy to manage costs
                        lifecycle_policy = {
                            'Rules': [
                                {
                                    'Id': 'DeleteOldVersions',
                                    'Status': 'Enabled',
                                    'NoncurrentVersionExpiration': {'NoncurrentDays': 30},
                                    'AbortIncompleteMultipartUpload': {'DaysAfterInitiation': 7}
                                }
                            ]
                        }
                        self.s3_client.put_bucket_lifecycle_configuration(
                            Bucket=bucket_name,
                            LifecycleConfiguration=lifecycle_policy
                        )
                        
                        print(f"  ✓ Created S3 bucket: {bucket_name}")
                        created.append(bucket_name)
                    except ClientError as create_error:
                        print(f"  ❌ Failed to create bucket {bucket_name}: {create_error}")
                else:
                    print(f"  ❌ Error checking bucket {bucket_name}: {e}")
        
        self.created_resources['s3_buckets'] = created
        return created
    
    def create_dynamodb_tables(self) -> List[str]:
        """Create DynamoDB tables for experiment tracking and model registry"""
        tables = [
            {
                'TableName': DYNAMODB_EXPERIMENTS_TABLE,
                'KeySchema': [
                    {'AttributeName': 'experiment_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'experiment_id', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'N'}
                ],
                'BillingMode': 'PAY_PER_REQUEST'  # Free tier friendly
            },
            {
                'TableName': DYNAMODB_MODELS_TABLE,
                'KeySchema': [
                    {'AttributeName': 'model_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'version', 'KeyType': 'RANGE'}
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'model_id', 'AttributeType': 'S'},
                    {'AttributeName': 'version', 'AttributeType': 'S'}
                ],
                'BillingMode': 'PAY_PER_REQUEST'
            },
            {
                'TableName': DYNAMODB_TRAINING_RUNS_TABLE,
                'KeySchema': [
                    {'AttributeName': 'run_id', 'KeyType': 'HASH'}
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'run_id', 'AttributeType': 'S'}
                ],
                'BillingMode': 'PAY_PER_REQUEST'
            }
        ]
        
        created = []
        for table_config in tables:
            table_name = table_config['TableName']
            
            if self.dry_run:
                print(f"  [DRY RUN] Would create DynamoDB table: {table_name}")
                continue
            
            try:
                # Check if table exists
                self.dynamodb_client.describe_table(TableName=table_name)
                print(f"  ✓ DynamoDB table already exists: {table_name}")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    try:
                        self.dynamodb_client.create_table(**table_config)
                        print(f"  ✓ Created DynamoDB table: {table_name}")
                        created.append(table_name)
                        
                        # Wait for table to be active
                        waiter = self.dynamodb_client.get_waiter('table_exists')
                        waiter.wait(TableName=table_name)
                    except ClientError as create_error:
                        print(f"  ❌ Failed to create table {table_name}: {create_error}")
                else:
                    print(f"  ❌ Error checking table {table_name}: {e}")
        
        self.created_resources['dynamodb_tables'] = created
        return created
    
    def create_iam_roles(self) -> List[str]:
        """Create IAM roles for Lambda execution"""
        lambda_role_name = f"{PROJECT_NAME}-lambda-execution-role"
        
        if self.dry_run:
            print(f"  [DRY RUN] Would create IAM role: {lambda_role_name}")
            return []
        
        # Load policy document
        policy_path = Path(__file__).parent.parent / 'aws_config' / 'iam_policies' / 'lambda_execution_role_policy.json'
        with open(policy_path, 'r') as f:
            policy_document = json.load(f)
        
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            # Check if role exists
            self.iam_client.get_role(RoleName=lambda_role_name)
            print(f"  ✓ IAM role already exists: {lambda_role_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                try:
                    # Create role
                    self.iam_client.create_role(
                        RoleName=lambda_role_name,
                        AssumeRolePolicyDocument=json.dumps(trust_policy),
                        Description='Execution role for OCT MLOps Lambda functions'
                    )
                    
                    # Attach policy
                    self.iam_client.put_role_policy(
                        RoleName=lambda_role_name,
                        PolicyName=f"{PROJECT_NAME}-lambda-policy",
                        PolicyDocument=json.dumps(policy_document)
                    )
                    
                    print(f"  ✓ Created IAM role: {lambda_role_name}")
                    self.created_resources['iam_roles'].append(lambda_role_name)
                except ClientError as create_error:
                    print(f"  ❌ Failed to create IAM role: {create_error}")
        
        return self.created_resources['iam_roles']
    
    def setup_billing_alerts(self) -> bool:
        """Set up CloudWatch billing alerts"""
        alert_email = os.getenv('BILLING_ALERT_EMAIL')
        threshold = float(os.getenv('BILLING_THRESHOLD_USD', 5.0))
        
        if not alert_email or alert_email == 'your_email@example.com':
            print("  ⚠️  Skipping billing alerts (no email configured)")
            return False
        
        if self.dry_run:
            print(f"  [DRY RUN] Would create billing alert for ${threshold} to {alert_email}")
            return True
        
        try:
            # Note: Billing metrics are only available in us-east-1
            billing_cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
            
            billing_cloudwatch.put_metric_alarm(
                AlarmName=f'{PROJECT_NAME}-billing-alert',
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=1,
                MetricName='EstimatedCharges',
                Namespace='AWS/Billing',
                Period=21600,  # 6 hours
                Statistic='Maximum',
                Threshold=threshold,
                ActionsEnabled=False,  # Set to True after creating SNS topic
                AlarmDescription=f'Alert when AWS charges exceed ${threshold}',
                Dimensions=[{'Name': 'Currency', 'Value': 'USD'}]
            )
            print(f"  ✓ Created billing alert (${threshold} threshold)")
            return True
        except Exception as e:
            print(f"  ⚠️  Could not create billing alert: {e}")
            return False
    
    def teardown(self):
        """Delete all created resources"""
        print("\n⚠️  WARNING: This will delete all MLOps resources!")
        print("Resources to delete:")
        print(f"  - S3 Buckets: {[S3_DATA_BUCKET, S3_MODELS_BUCKET, S3_ARTIFACTS_BUCKET]}")
        print(f"  - DynamoDB Tables: {[DYNAMODB_EXPERIMENTS_TABLE, DYNAMODB_MODELS_TABLE, DYNAMODB_TRAINING_RUNS_TABLE]}")
        
        if not self.dry_run:
            confirm = input("\nType 'DELETE' to confirm: ")
            if confirm != 'DELETE':
                print("Teardown cancelled.")
                return
        
        # Delete S3 buckets (must be empty first)
        for bucket_name in [S3_DATA_BUCKET, S3_MODELS_BUCKET, S3_ARTIFACTS_BUCKET]:
            if self.dry_run:
                print(f"  [DRY RUN] Would delete S3 bucket: {bucket_name}")
                continue
            
            try:
                # Delete all objects first
                bucket = boto3.resource('s3').Bucket(bucket_name)
                bucket.object_versions.all().delete()
                bucket.delete()
                print(f"  ✓ Deleted S3 bucket: {bucket_name}")
            except Exception as e:
                print(f"  ⚠️  Could not delete bucket {bucket_name}: {e}")
        
        # Delete DynamoDB tables
        for table_name in [DYNAMODB_EXPERIMENTS_TABLE, DYNAMODB_MODELS_TABLE, DYNAMODB_TRAINING_RUNS_TABLE]:
            if self.dry_run:
                print(f"  [DRY RUN] Would delete DynamoDB table: {table_name}")
                continue
            
            try:
                self.dynamodb_client.delete_table(TableName=table_name)
                print(f"  ✓ Deleted DynamoDB table: {table_name}")
            except Exception as e:
                print(f"  ⚠️  Could not delete table {table_name}: {e}")
        
        print("\n✓ Teardown complete!")
    
    def verify_setup(self) -> bool:
        """Verify all resources are created correctly"""
        print("\n🔍 Verifying setup...")
        all_good = True
        
        # Check S3 buckets
        for bucket_name in [S3_DATA_BUCKET, S3_MODELS_BUCKET, S3_ARTIFACTS_BUCKET]:
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
                print(f"  ✓ S3 bucket accessible: {bucket_name}")
            except:
                print(f"  ❌ S3 bucket not accessible: {bucket_name}")
                all_good = False
        
        # Check DynamoDB tables
        for table_name in [DYNAMODB_EXPERIMENTS_TABLE, DYNAMODB_MODELS_TABLE, DYNAMODB_TRAINING_RUNS_TABLE]:
            try:
                response = self.dynamodb_client.describe_table(TableName=table_name)
                status = response['Table']['TableStatus']
                print(f"  ✓ DynamoDB table {status}: {table_name}")
            except:
                print(f"  ❌ DynamoDB table not accessible: {table_name}")
                all_good = False
        
        return all_good


def main():
    parser = argparse.ArgumentParser(description='Setup AWS infrastructure for OCT MLOps')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without creating resources')
    parser.add_argument('--teardown', action='store_true', help='Delete all resources')
    parser.add_argument('--skip-iam', action='store_true', help='Skip IAM role creation (requires manual setup)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("OCT MLOps - AWS Infrastructure Setup")
    print("=" * 60)
    print(f"Project: {PROJECT_NAME}")
    print(f"Region: {AWS_REGION}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 60)
    
    setup = AWSSetup(dry_run=args.dry_run)
    
    if args.teardown:
        setup.teardown()
        return
    
    # Show cost estimate
    print("\n💰 Estimated Monthly Costs:")
    for service, cost in setup.estimate_costs().items():
        print(f"  {service}: {cost}")
    
    if not args.dry_run:
        print("\n⚠️  This will create AWS resources in your account.")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Setup cancelled.")
            return
    
    print("\n🚀 Starting setup...\n")
    
    # Create resources
    print("1️⃣  Creating S3 buckets...")
    setup.create_s3_buckets()
    
    print("\n2️⃣  Creating DynamoDB tables...")
    setup.create_dynamodb_tables()
    
    if not args.skip_iam:
        print("\n3️⃣  Creating IAM roles...")
        setup.create_iam_roles()
    else:
        print("\n3️⃣  Skipping IAM role creation (manual setup required)")
    
    print("\n4️⃣  Setting up billing alerts...")
    setup.setup_billing_alerts()
    
    if not args.dry_run:
        if setup.verify_setup():
            print("\n" + "=" * 60)
            print("✅ Setup complete! All resources created successfully.")
            print("=" * 60)
            print("\nNext steps:")
            print("1. Review created resources in AWS Console")
            print("2. Configure your training pipeline: mlops/config.py")
            print("3. Run experiment tracking: python mlops/experiment_tracker.py")
            print("4. Deploy Lambda functions: bash scripts/deploy_lambda.sh")
        else:
            print("\n⚠️  Setup completed with some errors. Check the output above.")
    else:
        print("\n" + "=" * 60)
        print("✅ Dry run complete! No resources were created.")
        print("=" * 60)
        print("\nTo actually create resources, run without --dry-run flag:")
        print("  python scripts/setup_aws.py")


if __name__ == '__main__':
    main()

