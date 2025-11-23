"""
MLOps Configuration Management
Centralized configuration for AWS resources and training parameters
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from dataclasses import dataclass, field

# Load AWS configuration
env_path = Path(__file__).parent.parent / 'aws_config' / '.env.aws'
if env_path.exists():
    load_dotenv(env_path)


@dataclass
class AWSConfig:
    """AWS resource configuration"""
    region: str = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    
    # S3 Buckets
    s3_data_bucket: str = os.getenv('S3_DATA_BUCKET', 'oct-mlops-data-dev')
    s3_models_bucket: str = os.getenv('S3_MODELS_BUCKET', 'oct-mlops-models-dev')
    s3_artifacts_bucket: str = os.getenv('S3_ARTIFACTS_BUCKET', 'oct-mlops-artifacts-dev')
    
    # DynamoDB Tables
    dynamodb_experiments_table: str = os.getenv('DYNAMODB_EXPERIMENTS_TABLE', 'oct-mlops-experiments')
    dynamodb_models_table: str = os.getenv('DYNAMODB_MODELS_TABLE', 'oct-mlops-model-registry')
    dynamodb_training_runs_table: str = os.getenv('DYNAMODB_TRAINING_RUNS_TABLE', 'oct-mlops-training-runs')
    
    # Lambda
    lambda_memory_size: int = int(os.getenv('LAMBDA_MEMORY_SIZE', '1024'))
    lambda_timeout: int = int(os.getenv('LAMBDA_TIMEOUT', '300'))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'region': self.region,
            's3': {
                'data': self.s3_data_bucket,
                'models': self.s3_models_bucket,
                'artifacts': self.s3_artifacts_bucket
            },
            'dynamodb': {
                'experiments': self.dynamodb_experiments_table,
                'models': self.dynamodb_models_table,
                'training_runs': self.dynamodb_training_runs_table
            }
        }


@dataclass
class ClassificationConfig:
    """Configuration for OCT classification model"""
    model_name: str = 'resnet50'  # resnet50, efficientnet_b0, etc.
    num_classes: int = 4
    class_names: list = field(default_factory=lambda: ['CNV', 'DME', 'DRUSEN', 'NORMAL'])
    image_size: tuple = (224, 224)
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    early_stopping_patience: int = 10
    weight_decay: float = 1e-5
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_strength: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'weight_decay': self.weight_decay,
            'use_augmentation': self.use_augmentation,
            'augmentation_strength': self.augmentation_strength
        }


@dataclass
class SegmentationConfig:
    """Configuration for OCT segmentation model"""
    model_name: str = 'unet'
    num_classes: int = 13
    class_names: list = field(default_factory=lambda: [
        'background', 'GCL', 'INL', 'IPL', 'ONL', 'OPL', 'RNFL', 'RPE',
        'CHOROID', 'INTRA-RETINAL-FLUID', 'SUB-RETINAL-FLUID', 'PED', 'DRUSENOID-PED'
    ])
    image_size: tuple = (256, 256)
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 75
    features: list = field(default_factory=lambda: [64, 128, 256, 512])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'features': self.features
        }


@dataclass
class MLOpsConfig:
    """Main MLOps configuration"""
    project_name: str = os.getenv('PROJECT_NAME', 'oct-mlops')
    environment: str = os.getenv('ENVIRONMENT', 'dev')
    
    # Component configs
    aws: AWSConfig = field(default_factory=AWSConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    
    # Paths
    local_data_dir: Path = Path('data')
    local_models_dir: Path = Path('classification_models')
    local_artifacts_dir: Path = Path('results')
    
    # Experiment tracking
    log_interval: int = 10  # Log every N batches
    save_checkpoint_interval: int = 5  # Save checkpoint every N epochs
    
    # Model versioning
    auto_register_models: bool = True
    model_tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.local_data_dir.mkdir(exist_ok=True)
        self.local_models_dir.mkdir(exist_ok=True)
        self.local_artifacts_dir.mkdir(exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'project_name': self.project_name,
            'environment': self.environment,
            'aws': self.aws.to_dict(),
            'classification': self.classification.to_dict(),
            'segmentation': self.segmentation.to_dict(),
            'paths': {
                'local_data': str(self.local_data_dir),
                'local_models': str(self.local_models_dir),
                'local_artifacts': str(self.local_artifacts_dir)
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MLOpsConfig':
        """Create config from dictionary"""
        config = cls()
        config.project_name = config_dict.get('project_name', config.project_name)
        config.environment = config_dict.get('environment', config.environment)
        return config


# Global config instance
config = MLOpsConfig()


def get_config() -> MLOpsConfig:
    """Get the global config instance"""
    return config


def update_config(**kwargs):
    """Update global config with new values"""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

