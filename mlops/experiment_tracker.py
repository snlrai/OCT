"""
Experiment Tracker
Track ML experiments using DynamoDB for metadata storage
"""

import boto3
import time
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from decimal import Decimal
from botocore.exceptions import ClientError

from .config import get_config


class DecimalEncoder(json.JSONEncoder):
    """Helper class to convert Decimal to float for JSON serialization"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


class ExperimentTracker:
    """
    Track machine learning experiments with metrics, parameters, and artifacts
    
    Usage:
        tracker = ExperimentTracker(experiment_name='oct-classification-v1')
        tracker.log_params({'learning_rate': 0.001, 'batch_size': 32})
        
        for epoch in range(num_epochs):
            # Training...
            tracker.log_metrics({'train_loss': loss, 'train_acc': acc}, step=epoch)
        
        tracker.log_artifact('model.pth', s3_path='s3://bucket/path/model.pth')
        tracker.end_experiment(status='completed')
    """
    
    def __init__(
        self,
        experiment_name: str,
        model_type: str = 'classification',
        tags: Optional[Dict[str, str]] = None,
        use_local: bool = False
    ):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of the experiment
            model_type: Type of model (classification or segmentation)
            tags: Additional tags for the experiment
            use_local: If True, store locally instead of DynamoDB (for testing)
        """
        self.config = get_config()
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.use_local = use_local
        
        # Generate unique IDs
        self.experiment_id = f"{experiment_name}-{uuid.uuid4().hex[:8]}"
        self.run_id = uuid.uuid4().hex
        self.start_time = time.time()
        
        # Initialize storage
        if not use_local:
            self.dynamodb = boto3.resource('dynamodb', region_name=self.config.aws.region)
            self.experiments_table = self.dynamodb.Table(self.config.aws.dynamodb_experiments_table)
            self.runs_table = self.dynamodb.Table(self.config.aws.dynamodb_training_runs_table)
        else:
            self.local_storage = {
                'experiment_id': self.experiment_id,
                'run_id': self.run_id,
                'params': {},
                'metrics': [],
                'artifacts': []
            }
        
        # Experiment metadata
        self.metadata = {
            'experiment_id': self.experiment_id,
            'experiment_name': experiment_name,
            'run_id': self.run_id,
            'model_type': model_type,
            'status': 'running',
            'start_time': self.start_time,
            'tags': tags or {},
            'params': {},
            'metrics': {},
            'artifacts': []
        }
        
        self._create_experiment()
    
    def _convert_to_dynamodb_format(self, obj: Any) -> Any:
        """Convert Python objects to DynamoDB compatible format"""
        if isinstance(obj, float):
            return Decimal(str(obj))
        elif isinstance(obj, dict):
            return {k: self._convert_to_dynamodb_format(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_dynamodb_format(v) for v in obj]
        return obj
    
    def _create_experiment(self):
        """Create experiment entry in DynamoDB"""
        if self.use_local:
            print(f"📝 [LOCAL] Created experiment: {self.experiment_id}")
            return
        
        try:
            item = {
                'experiment_id': self.experiment_id,
                'timestamp': Decimal(str(self.start_time)),
                'experiment_name': self.experiment_name,
                'run_id': self.run_id,
                'model_type': self.model_type,
                'status': 'running',
                'tags': json.dumps(self.metadata['tags']),
                'created_at': datetime.utcnow().isoformat()
            }
            
            self.experiments_table.put_item(Item=item)
            print(f"📝 Created experiment: {self.experiment_id}")
        except ClientError as e:
            print(f"⚠️  Warning: Could not create experiment in DynamoDB: {e}")
            self.use_local = True
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters for the experiment
        
        Args:
            params: Dictionary of hyperparameters
        """
        self.metadata['params'].update(params)
        
        if self.use_local:
            self.local_storage['params'].update(params)
            return
        
        try:
            self.experiments_table.update_item(
                Key={
                    'experiment_id': self.experiment_id,
                    'timestamp': Decimal(str(self.start_time))
                },
                UpdateExpression='SET params = :params',
                ExpressionAttributeValues={
                    ':params': json.dumps(self._convert_to_dynamodb_format(params), cls=DecimalEncoder)
                }
            )
        except ClientError as e:
            print(f"⚠️  Warning: Could not log params: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for the current step
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step/epoch number
        """
        timestamp = time.time()
        
        metric_entry = {
            'timestamp': timestamp,
            'step': step,
            'metrics': metrics
        }
        
        if self.use_local:
            self.local_storage['metrics'].append(metric_entry)
            return
        
        # Store in runs table for time-series metrics
        try:
            item = {
                'run_id': self.run_id,
                'timestamp': Decimal(str(timestamp)),
                'step': step if step is not None else 0,
                'metrics': self._convert_to_dynamodb_format(metrics)
            }
            self.runs_table.put_item(Item=item)
        except ClientError as e:
            print(f"⚠️  Warning: Could not log metrics: {e}")
    
    def log_artifact(self, artifact_name: str, artifact_path: str, artifact_type: str = 'model'):
        """
        Log artifact location
        
        Args:
            artifact_name: Name of the artifact
            artifact_path: S3 path or local path to artifact
            artifact_type: Type of artifact (model, plot, data, etc.)
        """
        artifact_entry = {
            'name': artifact_name,
            'path': artifact_path,
            'type': artifact_type,
            'created_at': datetime.utcnow().isoformat()
        }
        
        self.metadata['artifacts'].append(artifact_entry)
        
        if self.use_local:
            self.local_storage['artifacts'].append(artifact_entry)
            return
        
        try:
            # Update experiment with new artifact
            self.experiments_table.update_item(
                Key={
                    'experiment_id': self.experiment_id,
                    'timestamp': Decimal(str(self.start_time))
                },
                UpdateExpression='SET artifacts = list_append(if_not_exists(artifacts, :empty_list), :artifact)',
                ExpressionAttributeValues={
                    ':artifact': [artifact_entry],
                    ':empty_list': []
                }
            )
        except ClientError as e:
            print(f"⚠️  Warning: Could not log artifact: {e}")
    
    def end_experiment(self, status: str = 'completed', final_metrics: Optional[Dict[str, float]] = None):
        """
        End the experiment and log final status
        
        Args:
            status: Final status (completed, failed, stopped)
            final_metrics: Final metrics to log
        """
        end_time = time.time()
        duration = end_time - self.start_time
        
        if final_metrics:
            self.log_metrics(final_metrics, step=-1)
        
        if self.use_local:
            self.local_storage['status'] = status
            self.local_storage['duration'] = duration
            self._save_local()
            print(f"✅ Experiment ended ({status}) - Duration: {duration:.2f}s")
            return
        
        try:
            self.experiments_table.update_item(
                Key={
                    'experiment_id': self.experiment_id,
                    'timestamp': Decimal(str(self.start_time))
                },
                UpdateExpression='SET #status = :status, end_time = :end_time, duration = :duration',
                ExpressionAttributeNames={
                    '#status': 'status'
                },
                ExpressionAttributeValues={
                    ':status': status,
                    ':end_time': Decimal(str(end_time)),
                    ':duration': Decimal(str(duration))
                }
            )
            print(f"✅ Experiment ended ({status}) - Duration: {duration:.2f}s")
        except ClientError as e:
            print(f"⚠️  Warning: Could not end experiment: {e}")
    
    def _save_local(self):
        """Save experiment data locally"""
        output_dir = self.config.local_artifacts_dir / 'experiments'
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{self.experiment_id}.json"
        with open(output_file, 'w') as f:
            json.dump(self.local_storage, f, indent=2, cls=DecimalEncoder)
        
        print(f"💾 Saved experiment locally: {output_file}")
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get all metrics logged during this run"""
        if self.use_local:
            return self.local_storage.get('metrics', [])
        
        try:
            response = self.runs_table.query(
                KeyConditionExpression='run_id = :run_id',
                ExpressionAttributeValues={
                    ':run_id': self.run_id
                }
            )
            return response.get('Items', [])
        except ClientError as e:
            print(f"⚠️  Warning: Could not retrieve metrics: {e}")
            return []


def list_experiments(experiment_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    List all experiments, optionally filtered by name
    
    Args:
        experiment_name: Filter by experiment name
        limit: Maximum number of experiments to return
    
    Returns:
        List of experiment metadata dictionaries
    """
    config = get_config()
    dynamodb = boto3.resource('dynamodb', region_name=config.aws.region)
    table = dynamodb.Table(config.aws.dynamodb_experiments_table)
    
    try:
        if experiment_name:
            response = table.scan(
                FilterExpression='experiment_name = :name',
                ExpressionAttributeValues={':name': experiment_name},
                Limit=limit
            )
        else:
            response = table.scan(Limit=limit)
        
        return response.get('Items', [])
    except ClientError as e:
        print(f"⚠️  Warning: Could not list experiments: {e}")
        return []


def get_experiment(experiment_id: str) -> Optional[Dict[str, Any]]:
    """
    Get details of a specific experiment
    
    Args:
        experiment_id: ID of the experiment
    
    Returns:
        Experiment metadata dictionary
    """
    config = get_config()
    dynamodb = boto3.resource('dynamodb', region_name=config.aws.region)
    table = dynamodb.Table(config.aws.dynamodb_experiments_table)
    
    try:
        response = table.scan(
            FilterExpression='experiment_id = :id',
            ExpressionAttributeValues={':id': experiment_id},
            Limit=1
        )
        items = response.get('Items', [])
        return items[0] if items else None
    except ClientError as e:
        print(f"⚠️  Warning: Could not get experiment: {e}")
        return None

