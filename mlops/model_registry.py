"""
Model Registry
Manage model versions, metadata, and artifacts using S3 and DynamoDB
"""

import boto3
import json
import uuid
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from decimal import Decimal
from botocore.exceptions import ClientError
import hashlib

from .config import get_config


class ModelRegistry:
    """
    Centralized model registry for versioning and managing ML models
    
    Usage:
        registry = ModelRegistry()
        
        # Register a new model
        model_info = registry.register_model(
            model_name='oct-classifier',
            model_path='classification_models/best_oct_classifier.pth',
            model_type='classification',
            metrics={'accuracy': 0.95, 'f1_score': 0.93},
            metadata={'framework': 'pytorch', 'architecture': 'resnet50'}
        )
        
        # Get latest model
        latest = registry.get_latest_model('oct-classifier')
        
        # Download model
        local_path = registry.download_model('oct-classifier', version='v1')
    """
    
    def __init__(self, use_local: bool = False):
        """
        Initialize model registry
        
        Args:
            use_local: If True, store locally instead of AWS (for testing)
        """
        self.config = get_config()
        self.use_local = use_local
        
        if not use_local:
            self.s3_client = boto3.client('s3', region_name=self.config.aws.region)
            self.dynamodb = boto3.resource('dynamodb', region_name=self.config.aws.region)
            self.models_table = self.dynamodb.Table(self.config.aws.dynamodb_models_table)
            self.models_bucket = self.config.aws.s3_models_bucket
        else:
            self.local_registry_dir = Path('local_model_registry')
            self.local_registry_dir.mkdir(exist_ok=True)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_next_version(self, model_name: str) -> str:
        """Get the next version number for a model"""
        if self.use_local:
            existing = list(self.local_registry_dir.glob(f"{model_name}_v*.json"))
            if not existing:
                return "v1"
            versions = [int(p.stem.split('_v')[1]) for p in existing]
            return f"v{max(versions) + 1}"
        
        try:
            response = self.models_table.query(
                KeyConditionExpression='model_id = :model_id',
                ExpressionAttributeValues={':model_id': model_name},
                ScanIndexForward=False,  # Sort descending
                Limit=1
            )
            
            if response.get('Items'):
                last_version = response['Items'][0]['version']
                version_num = int(last_version.replace('v', ''))
                return f"v{version_num + 1}"
            return "v1"
        except ClientError:
            return "v1"
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        experiment_id: Optional[str] = None,
        auto_version: bool = True
    ) -> Dict[str, Any]:
        """
        Register a new model version
        
        Args:
            model_name: Name of the model (e.g., 'oct-classifier')
            model_path: Local path to the model file
            model_type: Type of model ('classification' or 'segmentation')
            metrics: Dictionary of model performance metrics
            metadata: Additional metadata about the model
            tags: Tags for categorization
            experiment_id: Associated experiment ID
            auto_version: Automatically assign version number
        
        Returns:
            Dictionary with model registration details
        """
        # Validate model file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Get version
        version = self._get_next_version(model_name) if auto_version else "v1"
        
        # Calculate file hash for integrity
        file_hash = self._calculate_file_hash(model_path)
        file_size = os.path.getsize(model_path)
        
        # Upload to S3 or save locally
        if self.use_local:
            s3_path = self._save_model_locally(model_name, version, model_path)
        else:
            s3_path = self._upload_to_s3(model_name, version, model_path)
        
        # Prepare model metadata
        model_info = {
            'model_id': model_name,
            'version': version,
            'model_type': model_type,
            'metrics': metrics,
            's3_path': s3_path,
            'file_hash': file_hash,
            'file_size': file_size,
            'metadata': metadata or {},
            'tags': tags or {},
            'experiment_id': experiment_id,
            'status': 'active',
            'created_at': datetime.utcnow().isoformat(),
            'created_by': os.getenv('USER', 'unknown')
        }
        
        # Store metadata
        if self.use_local:
            self._save_metadata_locally(model_info)
        else:
            self._save_metadata_to_dynamodb(model_info)
        
        print(f"✅ Registered model: {model_name} {version}")
        print(f"   Path: {s3_path}")
        print(f"   Metrics: {metrics}")
        
        return model_info
    
    def _upload_to_s3(self, model_name: str, version: str, model_path: str) -> str:
        """Upload model to S3"""
        s3_key = f"models/{model_name}/{version}/{Path(model_path).name}"
        
        try:
            self.s3_client.upload_file(
                model_path,
                self.models_bucket,
                s3_key,
                ExtraArgs={'Metadata': {'version': version, 'model_name': model_name}}
            )
            s3_path = f"s3://{self.models_bucket}/{s3_key}"
            print(f"📤 Uploaded model to: {s3_path}")
            return s3_path
        except ClientError as e:
            print(f"❌ Failed to upload model to S3: {e}")
            raise
    
    def _save_model_locally(self, model_name: str, version: str, model_path: str) -> str:
        """Save model copy locally"""
        local_dir = self.local_registry_dir / model_name / version
        local_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        dest_path = local_dir / Path(model_path).name
        shutil.copy2(model_path, dest_path)
        
        return str(dest_path)
    
    def _save_metadata_to_dynamodb(self, model_info: Dict[str, Any]):
        """Save model metadata to DynamoDB"""
        # Convert floats to Decimal for DynamoDB
        def convert_decimals(obj):
            if isinstance(obj, float):
                return Decimal(str(obj))
            elif isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals(v) for v in obj]
            return obj
        
        item = convert_decimals(model_info)
        
        try:
            self.models_table.put_item(Item=item)
        except ClientError as e:
            print(f"❌ Failed to save metadata to DynamoDB: {e}")
            raise
    
    def _save_metadata_locally(self, model_info: Dict[str, Any]):
        """Save model metadata locally"""
        metadata_file = self.local_registry_dir / f"{model_info['model_id']}_{model_info['version']}.json"
        with open(metadata_file, 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def get_model(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get model metadata
        
        Args:
            model_name: Name of the model
            version: Specific version (or latest if None)
        
        Returns:
            Model metadata dictionary
        """
        if version is None:
            return self.get_latest_model(model_name)
        
        if self.use_local:
            metadata_file = self.local_registry_dir / f"{model_name}_{version}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            return None
        
        try:
            response = self.models_table.get_item(
                Key={'model_id': model_name, 'version': version}
            )
            return response.get('Item')
        except ClientError as e:
            print(f"⚠️  Could not retrieve model: {e}")
            return None
    
    def get_latest_model(self, model_name: str, status: str = 'active') -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a model
        
        Args:
            model_name: Name of the model
            status: Filter by status (active, archived, etc.)
        
        Returns:
            Model metadata dictionary
        """
        if self.use_local:
            existing = sorted(
                self.local_registry_dir.glob(f"{model_name}_v*.json"),
                key=lambda p: int(p.stem.split('_v')[1]),
                reverse=True
            )
            if existing:
                with open(existing[0], 'r') as f:
                    return json.load(f)
            return None
        
        try:
            response = self.models_table.query(
                KeyConditionExpression='model_id = :model_id',
                ExpressionAttributeValues={':model_id': model_name},
                ScanIndexForward=False,
                Limit=10
            )
            
            # Filter by status
            items = [item for item in response.get('Items', []) if item.get('status') == status]
            return items[0] if items else None
        except ClientError as e:
            print(f"⚠️  Could not retrieve latest model: {e}")
            return None
    
    def download_model(self, model_name: str, version: Optional[str] = None, output_dir: Optional[str] = None) -> str:
        """
        Download model file from S3 to local directory
        
        Args:
            model_name: Name of the model
            version: Version to download (latest if None)
            output_dir: Local directory to save model
        
        Returns:
            Path to downloaded model file
        """
        model_info = self.get_model(model_name, version)
        if not model_info:
            raise ValueError(f"Model not found: {model_name} {version or 'latest'}")
        
        s3_path = model_info['s3_path']
        
        if self.use_local:
            # Already local, just return path
            return s3_path
        
        # Parse S3 path
        if not s3_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path: {s3_path}")
        
        parts = s3_path.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1]
        
        # Set output path
        if output_dir is None:
            output_dir = self.config.local_models_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(key).name
        
        try:
            self.s3_client.download_file(bucket, key, str(output_path))
            print(f"📥 Downloaded model to: {output_path}")
            return str(output_path)
        except ClientError as e:
            print(f"❌ Failed to download model: {e}")
            raise
    
    def list_models(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered models
        
        Args:
            model_name: Filter by model name (all if None)
        
        Returns:
            List of model metadata dictionaries
        """
        if self.use_local:
            if model_name:
                files = self.local_registry_dir.glob(f"{model_name}_v*.json")
            else:
                files = self.local_registry_dir.glob("*.json")
            
            models = []
            for f in files:
                with open(f, 'r') as file:
                    models.append(json.load(file))
            return sorted(models, key=lambda x: x['created_at'], reverse=True)
        
        try:
            if model_name:
                response = self.models_table.query(
                    KeyConditionExpression='model_id = :model_id',
                    ExpressionAttributeValues={':model_id': model_name}
                )
            else:
                response = self.models_table.scan()
            
            return response.get('Items', [])
        except ClientError as e:
            print(f"⚠️  Could not list models: {e}")
            return []
    
    def update_model_status(self, model_name: str, version: str, status: str):
        """
        Update model status (active, archived, deprecated)
        
        Args:
            model_name: Name of the model
            version: Version to update
            status: New status
        """
        if self.use_local:
            model_info = self.get_model(model_name, version)
            if model_info:
                model_info['status'] = status
                self._save_metadata_locally(model_info)
            return
        
        try:
            self.models_table.update_item(
                Key={'model_id': model_name, 'version': version},
                UpdateExpression='SET #status = :status, updated_at = :updated_at',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':status': status,
                    ':updated_at': datetime.utcnow().isoformat()
                }
            )
            print(f"✅ Updated {model_name} {version} status to: {status}")
        except ClientError as e:
            print(f"❌ Failed to update model status: {e}")
    
    def compare_models(self, model_name: str, versions: List[str]) -> Dict[str, Any]:
        """
        Compare multiple versions of a model
        
        Args:
            model_name: Name of the model
            versions: List of versions to compare
        
        Returns:
            Comparison dictionary with metrics for each version
        """
        comparison = {}
        for version in versions:
            model_info = self.get_model(model_name, version)
            if model_info:
                comparison[version] = {
                    'metrics': model_info.get('metrics', {}),
                    'created_at': model_info.get('created_at'),
                    'file_size': model_info.get('file_size'),
                    'status': model_info.get('status')
                }
        
        return comparison

