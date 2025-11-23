"""
CloudWatch Monitoring for OCT MLOps
Track metrics, logs, and system health
"""

import boto3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal

from .config import get_config


class CloudWatchMonitor:
    """Monitor and log metrics to CloudWatch"""
    
    def __init__(self, namespace: str = 'OCT-MLOps'):
        self.config = get_config()
        self.namespace = namespace
        self.cloudwatch = boto3.client('cloudwatch', region_name=self.config.aws.region)
        self.logs = boto3.client('logs', region_name=self.config.aws.region)
        
        # Create log group if it doesn't exist
        self.log_group_name = f"/aws/mlops/{self.config.project_name}"
        self._ensure_log_group_exists()
    
    def _ensure_log_group_exists(self):
        """Create CloudWatch log group if it doesn't exist"""
        try:
            self.logs.create_log_group(logGroupName=self.log_group_name)
            print(f"Created log group: {self.log_group_name}")
            
            # Set retention policy (7 days for free tier)
            self.logs.put_retention_policy(
                logGroupName=self.log_group_name,
                retentionInDays=7
            )
        except self.logs.exceptions.ResourceAlreadyExistsException:
            pass
    
    def put_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = 'None',
        dimensions: Optional[Dict[str, str]] = None
    ):
        """
        Put a custom metric to CloudWatch
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            dimensions: Dimensions for the metric
        """
        metric_data = {
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': datetime.utcnow()
        }
        
        if dimensions:
            metric_data['Dimensions'] = [
                {'Name': k, 'Value': v} for k, v in dimensions.items()
            ]
        
        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[metric_data]
            )
        except Exception as e:
            print(f"Warning: Failed to put metric {metric_name}: {e}")
    
    def log_inference_metrics(
        self,
        model_type: str,
        latency_ms: float,
        confidence: float,
        success: bool = True
    ):
        """Log inference metrics"""
        dimensions = {'ModelType': model_type}
        
        self.put_metric('InferenceLatency', latency_ms, 'Milliseconds', dimensions)
        self.put_metric('InferenceConfidence', confidence, 'None', dimensions)
        self.put_metric('InferenceSuccess', 1 if success else 0, 'Count', dimensions)
    
    def log_training_metrics(
        self,
        experiment_id: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float
    ):
        """Log training metrics"""
        dimensions = {'ExperimentID': experiment_id, 'Epoch': str(epoch)}
        
        self.put_metric('TrainLoss', train_loss, 'None', dimensions)
        self.put_metric('ValLoss', val_loss, 'None', dimensions)
        self.put_metric('TrainAccuracy', train_acc, 'Percent', dimensions)
        self.put_metric('ValAccuracy', val_acc, 'Percent', dimensions)
    
    def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        stat: str = 'Average',
        period: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Retrieve metrics from CloudWatch
        
        Args:
            metric_name: Name of the metric
            start_time: Start time for metrics
            end_time: End time for metrics
            stat: Statistic to retrieve (Average, Sum, Maximum, Minimum)
            period: Period in seconds
        
        Returns:
            List of metric datapoints
        """
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace=self.namespace,
                MetricName=metric_name,
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=[stat]
            )
            return response.get('Datapoints', [])
        except Exception as e:
            print(f"Warning: Failed to get metrics: {e}")
            return []
    
    def create_dashboard(self):
        """Create CloudWatch dashboard for OCT MLOps"""
        dashboard_name = f"{self.config.project_name}-dashboard"
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            [self.namespace, "InferenceLatency", {"stat": "Average"}],
                            ["...", {"stat": "Maximum"}]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.config.aws.region,
                        "title": "Inference Latency",
                        "yAxis": {"left": {"label": "ms"}}
                    }
                },
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            [self.namespace, "InferenceSuccess", {"stat": "Sum"}]
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": self.config.aws.region,
                        "title": "Inference Requests",
                        "yAxis": {"left": {"label": "Count"}}
                    }
                },
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            [self.namespace, "InferenceConfidence", {"stat": "Average"}]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.config.aws.region,
                        "title": "Model Confidence",
                        "yAxis": {"left": {"min": 0, "max": 1}}
                    }
                },
                {
                    "type": "log",
                    "properties": {
                        "query": f"SOURCE '{self.log_group_name}' | fields @timestamp, @message | sort @timestamp desc | limit 20",
                        "region": self.config.aws.region,
                        "title": "Recent Logs"
                    }
                }
            ]
        }
        
        try:
            self.cloudwatch.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=str(dashboard_body)
            )
            print(f"✓ Created CloudWatch dashboard: {dashboard_name}")
            print(f"  View at: https://console.aws.amazon.com/cloudwatch/home?region={self.config.aws.region}#dashboards:name={dashboard_name}")
        except Exception as e:
            print(f"Warning: Failed to create dashboard: {e}")


class CostMonitor:
    """Monitor AWS costs and usage"""
    
    def __init__(self):
        self.config = get_config()
        self.ce_client = boto3.client('ce', region_name='us-east-1')  # Cost Explorer is global
        self.cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')  # Billing metrics in us-east-1
    
    def get_current_month_costs(self) -> Dict[str, Any]:
        """Get costs for the current month"""
        start_date = datetime.now().replace(day=1).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={'Start': start_date, 'End': end_date},
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                GroupBy=[{'Type': 'SERVICE', 'Key': 'SERVICE'}]
            )
            
            costs = {}
            total_cost = 0.0
            
            for result in response['ResultsByTime']:
                for group in result['Groups']:
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    costs[service] = cost
                    total_cost += cost
            
            return {
                'total': total_cost,
                'by_service': costs,
                'period': f"{start_date} to {end_date}"
            }
        except Exception as e:
            print(f"Warning: Failed to get cost data: {e}")
            return {'total': 0.0, 'by_service': {}, 'error': str(e)}
    
    def get_service_usage(self, service: str) -> Dict[str, Any]:
        """Get usage statistics for a specific AWS service"""
        # Service-specific metrics
        service_metrics = {
            's3': self._get_s3_usage(),
            'dynamodb': self._get_dynamodb_usage(),
            'lambda': self._get_lambda_usage(),
            'ec2': self._get_ec2_usage()
        }
        
        return service_metrics.get(service.lower(), {})
    
    def _get_s3_usage(self) -> Dict[str, Any]:
        """Get S3 bucket usage"""
        s3 = boto3.client('s3', region_name=self.config.aws.region)
        cloudwatch = boto3.client('cloudwatch', region_name=self.config.aws.region)
        
        buckets = [
            self.config.aws.s3_data_bucket,
            self.config.aws.s3_models_bucket,
            self.config.aws.s3_artifacts_bucket
        ]
        
        usage = {}
        for bucket in buckets:
            try:
                # Get bucket size
                response = cloudwatch.get_metric_statistics(
                    Namespace='AWS/S3',
                    MetricName='BucketSizeBytes',
                    Dimensions=[
                        {'Name': 'BucketName', 'Value': bucket},
                        {'Name': 'StorageType', 'Value': 'StandardStorage'}
                    ],
                    StartTime=datetime.now() - timedelta(days=1),
                    EndTime=datetime.now(),
                    Period=86400,
                    Statistics=['Average']
                )
                
                if response['Datapoints']:
                    size_bytes = response['Datapoints'][0]['Average']
                    size_mb = size_bytes / (1024 * 1024)
                    usage[bucket] = f"{size_mb:.2f} MB"
            except Exception as e:
                usage[bucket] = f"Error: {e}"
        
        return usage
    
    def _get_dynamodb_usage(self) -> Dict[str, Any]:
        """Get DynamoDB table usage"""
        dynamodb = boto3.client('dynamodb', region_name=self.config.aws.region)
        
        tables = [
            self.config.aws.dynamodb_experiments_table,
            self.config.aws.dynamodb_models_table,
            self.config.aws.dynamodb_training_runs_table
        ]
        
        usage = {}
        for table in tables:
            try:
                response = dynamodb.describe_table(TableName=table)
                table_size = response['Table']['TableSizeBytes']
                item_count = response['Table']['ItemCount']
                usage[table] = {
                    'size_mb': f"{table_size / (1024 * 1024):.2f}",
                    'item_count': item_count
                }
            except Exception as e:
                usage[table] = f"Error: {e}"
        
        return usage
    
    def _get_lambda_usage(self) -> Dict[str, Any]:
        """Get Lambda function usage"""
        lambda_client = boto3.client('lambda', region_name=self.config.aws.region)
        
        function_name = f"{self.config.project_name}-inference"
        
        try:
            # Get invocation metrics
            cloudwatch = boto3.client('cloudwatch', region_name=self.config.aws.region)
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Invocations',
                Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                StartTime=datetime.now() - timedelta(days=1),
                EndTime=datetime.now(),
                Period=86400,
                Statistics=['Sum']
            )
            
            invocations = 0
            if response['Datapoints']:
                invocations = int(response['Datapoints'][0]['Sum'])
            
            return {
                'function': function_name,
                'invocations_24h': invocations
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_ec2_usage(self) -> Dict[str, Any]:
        """Get EC2 instance usage"""
        ec2 = boto3.client('ec2', region_name=self.config.aws.region)
        
        try:
            response = ec2.describe_instances(
                Filters=[
                    {'Name': 'tag:Project', 'Values': [self.config.project_name]}
                ]
            )
            
            instances = []
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instances.append({
                        'id': instance['InstanceId'],
                        'type': instance['InstanceType'],
                        'state': instance['State']['Name']
                    })
            
            return {'instances': instances}
        except Exception as e:
            return {'error': str(e)}
    
    def print_cost_report(self):
        """Print a formatted cost report"""
        print("\n" + "=" * 60)
        print("AWS Cost Report - OCT MLOps")
        print("=" * 60)
        
        costs = self.get_current_month_costs()
        
        print(f"\nTotal Cost (MTD): ${costs['total']:.2f}")
        print(f"Period: {costs.get('period', 'N/A')}")
        
        if costs['by_service']:
            print("\nCosts by Service:")
            for service, cost in sorted(costs['by_service'].items(), key=lambda x: x[1], reverse=True):
                if cost > 0:
                    print(f"  {service}: ${cost:.2f}")
        
        print("\n" + "-" * 60)
        print("Resource Usage:")
        print("-" * 60)
        
        # S3 usage
        s3_usage = self.get_service_usage('s3')
        if s3_usage:
            print("\nS3 Buckets:")
            for bucket, size in s3_usage.items():
                print(f"  {bucket}: {size}")
        
        # DynamoDB usage
        dynamo_usage = self.get_service_usage('dynamodb')
        if dynamo_usage:
            print("\nDynamoDB Tables:")
            for table, stats in dynamo_usage.items():
                if isinstance(stats, dict):
                    print(f"  {table}: {stats['size_mb']} MB, {stats['item_count']} items")
        
        # Lambda usage
        lambda_usage = self.get_service_usage('lambda')
        if lambda_usage and 'invocations_24h' in lambda_usage:
            print(f"\nLambda Invocations (24h): {lambda_usage['invocations_24h']}")
        
        print("\n" + "=" * 60)
        
        # Warning if approaching free tier limits
        if costs['total'] > 1.0:
            print("⚠️  Warning: Costs exceeding $1.00. Review usage to stay in free tier.")
        else:
            print("✓ Within expected free tier limits")
        
        print("=" * 60 + "\n")


if __name__ == '__main__':
    # Test monitoring
    cost_monitor = CostMonitor()
    cost_monitor.print_cost_report()

