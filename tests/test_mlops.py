"""
Unit tests for MLOps components
Run with: pytest tests/test_mlops.py -v
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn as nn

from mlops.config import MLOpsConfig, AWSConfig, ClassificationConfig
from mlops.experiment_tracker import ExperimentTracker
from mlops.model_registry import ModelRegistry


class TestConfig:
    """Test configuration management"""
    
    def test_aws_config_defaults(self):
        config = AWSConfig()
        assert config.region is not None
        assert 'oct-mlops' in config.s3_data_bucket.lower()
    
    def test_classification_config(self):
        config = ClassificationConfig()
        assert config.num_classes == 4
        assert len(config.class_names) == 4
        assert config.image_size == (224, 224)
    
    def test_mlops_config_to_dict(self):
        config = MLOpsConfig()
        config_dict = config.to_dict()
        assert 'aws' in config_dict
        assert 'classification' in config_dict
        assert 'project_name' in config_dict


class TestExperimentTracker:
    """Test experiment tracking"""
    
    def test_create_experiment_local(self):
        tracker = ExperimentTracker(
            experiment_name='test-exp',
            model_type='classification',
            use_local=True
        )
        assert tracker.experiment_id is not None
        assert tracker.run_id is not None
        assert 'test-exp' in tracker.experiment_id
    
    def test_log_params(self):
        tracker = ExperimentTracker(
            experiment_name='test-params',
            use_local=True
        )
        
        params = {'learning_rate': 0.001, 'batch_size': 32}
        tracker.log_params(params)
        
        assert tracker.metadata['params'] == params
    
    def test_log_metrics(self):
        tracker = ExperimentTracker(
            experiment_name='test-metrics',
            use_local=True
        )
        
        metrics = {'loss': 0.5, 'accuracy': 0.85}
        tracker.log_metrics(metrics, step=1)
        
        assert len(tracker.local_storage['metrics']) == 1
    
    def test_log_artifact(self):
        tracker = ExperimentTracker(
            experiment_name='test-artifact',
            use_local=True
        )
        
        tracker.log_artifact('model.pth', '/path/to/model.pth', 'model')
        assert len(tracker.metadata['artifacts']) == 1
        assert tracker.metadata['artifacts'][0]['name'] == 'model.pth'
    
    def test_end_experiment(self):
        tracker = ExperimentTracker(
            experiment_name='test-end',
            use_local=True
        )
        
        tracker.end_experiment(status='completed')
        assert tracker.local_storage['status'] == 'completed'


class TestModelRegistry:
    """Test model registry"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def dummy_model(self, temp_dir):
        """Create a dummy model file"""
        model_path = temp_dir / 'test_model.pth'
        
        # Create a simple model
        model = nn.Linear(10, 4)
        torch.save(model.state_dict(), model_path)
        
        return str(model_path)
    
    def test_register_model(self, dummy_model, temp_dir):
        # Change local registry dir to temp dir
        registry = ModelRegistry(use_local=True)
        registry.local_registry_dir = temp_dir
        
        model_info = registry.register_model(
            model_name='test-model',
            model_path=dummy_model,
            model_type='classification',
            metrics={'accuracy': 0.95}
        )
        
        assert model_info['model_id'] == 'test-model'
        assert model_info['version'] == 'v1'
        assert model_info['metrics']['accuracy'] == 0.95
    
    def test_get_latest_model(self, dummy_model, temp_dir):
        registry = ModelRegistry(use_local=True)
        registry.local_registry_dir = temp_dir
        
        # Register first model
        registry.register_model(
            model_name='test-model',
            model_path=dummy_model,
            model_type='classification',
            metrics={'accuracy': 0.90}
        )
        
        # Get latest
        latest = registry.get_latest_model('test-model')
        assert latest is not None
        assert latest['version'] == 'v1'
    
    def test_list_models(self, dummy_model, temp_dir):
        registry = ModelRegistry(use_local=True)
        registry.local_registry_dir = temp_dir
        
        # Register multiple models
        for i in range(3):
            registry.register_model(
                model_name='test-model',
                model_path=dummy_model,
                model_type='classification',
                metrics={'accuracy': 0.90 + i * 0.01}
            )
        
        models = registry.list_models('test-model')
        assert len(models) == 3
    
    def test_version_auto_increment(self, dummy_model, temp_dir):
        registry = ModelRegistry(use_local=True)
        registry.local_registry_dir = temp_dir
        
        # Register first model
        info1 = registry.register_model(
            model_name='test-model',
            model_path=dummy_model,
            model_type='classification',
            metrics={'accuracy': 0.90}
        )
        
        # Register second model
        info2 = registry.register_model(
            model_name='test-model',
            model_path=dummy_model,
            model_type='classification',
            metrics={'accuracy': 0.92}
        )
        
        assert info1['version'] == 'v1'
        assert info2['version'] == 'v2'


class TestIntegration:
    """Integration tests"""
    
    def test_experiment_to_model_registry(self, tmp_path):
        """Test full workflow: experiment → model registry"""
        
        # Create experiment
        tracker = ExperimentTracker(
            experiment_name='integration-test',
            use_local=True
        )
        
        tracker.log_params({'epochs': 10, 'lr': 0.001})
        tracker.log_metrics({'val_acc': 0.95}, step=10)
        
        # Create dummy model
        model_path = tmp_path / 'model.pth'
        model = nn.Linear(10, 4)
        torch.save(model.state_dict(), model_path)
        
        # Register model
        registry = ModelRegistry(use_local=True)
        registry.local_registry_dir = tmp_path
        
        model_info = registry.register_model(
            model_name='test-integration',
            model_path=str(model_path),
            model_type='classification',
            metrics={'val_acc': 0.95},
            experiment_id=tracker.experiment_id
        )
        
        tracker.log_artifact('model', model_info['s3_path'], 'model')
        tracker.end_experiment(status='completed')
        
        # Verify
        assert model_info['experiment_id'] == tracker.experiment_id
        assert len(tracker.metadata['artifacts']) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

