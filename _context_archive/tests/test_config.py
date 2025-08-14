"""
Unit tests for ConfigurationLoader class
========================================

Tests configuration loading and priority system.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import patch, mock_open
from app.config import ConfigurationLoader


class TestConfigurationLoader:
    """Test cases for configuration management"""
    
    def test_default_config_loading(self):
        """Test loading of default configuration values"""
        with patch('os.path.exists', return_value=False):
            config = ConfigurationLoader()
            
            # Test default model configuration
            model_path = config.get("model", "path")
            assert "phi3-128k-npu" in model_path
            assert config.get("model", "type") == "phi3"
            
            # Test default deployment settings
            assert config.get("deployment", "target_device") == "NPU"
            assert config.get("deployment", "npu_profile") == "balanced"
    
    def test_json_config_loading(self):
        """Test loading configuration from JSON file"""
        test_config = {
            "model": {
                "path": "/test/model/path",
                "name": "Test-Model",
                "type": "test"
            },
            "deployment": {
                "target_device": "CPU",
                "npu_profile": "conservative"
            }
        }
        
        config_content = json.dumps(test_config)
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=config_content)):
                config = ConfigurationLoader()
                
                assert config.get("model", "path") == "/test/model/path"
                assert config.get("model", "type") == "test"
                assert config.get("deployment", "target_device") == "CPU"
    
    @patch.dict(os.environ, {
        'MODEL_PATH': '/env/model/path',
        'TARGET_DEVICE': 'GPU',
        'NPU_PROFILE': 'aggressive'
    })
    def test_environment_override(self):
        """Test environment variable override"""
        with patch('os.path.exists', return_value=False):
            config = ConfigurationLoader()
            
            # Environment variables should override defaults
            assert config.get("model", "path") == "/env/model/path"
            assert config.get("deployment", "target_device") == "GPU"
            assert config.get("deployment", "npu_profile") == "aggressive"
    
    @patch.dict(os.environ, {
        'QWEN3_MODEL_PATH': '/legacy/model/path'
    })
    def test_legacy_env_variable_support(self):
        """Test backward compatibility with legacy environment variables"""
        with patch('os.path.exists', return_value=False):
            config = ConfigurationLoader()
            
            # Legacy QWEN3_MODEL_PATH should still work
            assert config.get("model", "path") == "/legacy/model/path"
    
    @patch.dict(os.environ, {
        'MODEL_PATH': '/new/model/path',
        'QWEN3_MODEL_PATH': '/legacy/model/path'
    })
    def test_env_variable_priority(self):
        """Test that new environment variables take precedence over legacy ones"""
        with patch('os.path.exists', return_value=False):
            config = ConfigurationLoader()
            
            # New MODEL_PATH should override legacy QWEN3_MODEL_PATH
            assert config.get("model", "path") == "/new/model/path"
    
    def test_nested_config_access(self):
        """Test nested configuration value access"""
        test_config = {
            "ui": {
                "performance": {
                    "timeout": 45.0
                }
            }
        }
        
        config_content = json.dumps(test_config)
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=config_content)):
                config = ConfigurationLoader()
                
                # Test nested access
                section = config.get_section("ui")
                assert section["performance"]["timeout"] == 45.0
    
    def test_missing_config_fallback(self):
        """Test fallback when configuration keys are missing"""
        with patch('os.path.exists', return_value=False):
            config = ConfigurationLoader()
            
            # Should return default for missing keys
            missing_value = config.get("nonexistent", "key", "default_value")
            assert missing_value == "default_value"
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON configuration"""
        invalid_json = "{ invalid json content"
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=invalid_json)):
                # Should fall back to defaults without crashing
                config = ConfigurationLoader()
                
                # Should still have default values
                assert config.get("model", "type") == "phi3"
    
    def test_configuration_priority_order(self):
        """Test the 4-tier configuration priority system"""
        # Setup: JSON config with base values
        json_config = {
            "model": {"path": "/json/path"},
            "deployment": {"target_device": "CPU"}
        }
        
        config_content = json.dumps(json_config)
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=config_content)):
                with patch.dict(os.environ, {'MODEL_PATH': '/env/path'}):
                    config = ConfigurationLoader()
                    
                    # ENV should override JSON
                    assert config.get("model", "path") == "/env/path"
                    # JSON should override defaults
                    assert config.get("deployment", "target_device") == "CPU"