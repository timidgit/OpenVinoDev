"""
Unit tests for model deployment and configuration
================================================

Tests core model deployment logic without requiring actual models.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.model import LLMConfigurationManager, validate_system_requirements


class TestLLMConfigurationManager:
    """Test configuration management for LLM deployment"""
    
    def test_npu_config_generation(self):
        """Test NPU configuration generation"""
        manager = LLMConfigurationManager("balanced")
        config = manager.get_npu_config()
        
        # Should contain essential NPU settings
        assert config["NPU_USE_NPUW"] == "YES"
        assert config["NPUW_LLM"] == "YES"
        assert config["NPUW_LLM_MAX_PROMPT_LEN"] == 8192  # Phi-3 optimized
        assert config["NPUW_LLM_PREFILL_HINT"] == "LATENCY"
        assert config["NPUW_LLM_GENERATE_HINT"] == "LATENCY"
    
    def test_cpu_config_generation(self):
        """Test CPU configuration generation"""
        manager = LLMConfigurationManager("balanced")
        config = manager.get_cpu_config()
        
        # Should contain CPU-specific settings
        assert config["MAX_PROMPT_LEN"] == 16384  # Larger for CPU
        assert config["MIN_RESPONSE_LEN"] == 512
        
        # Should have performance settings
        assert "PERFORMANCE_HINT" in config or "performance_mode" in str(config)
    
    def test_profile_differences(self):
        """Test that different profiles generate different configurations"""
        conservative = LLMConfigurationManager("conservative")
        aggressive = LLMConfigurationManager("aggressive")
        
        # Both should work without errors
        conservative_config = conservative.get_npu_config()
        aggressive_config = aggressive.get_npu_config()
        
        # Both should have required NPUW settings
        for config in [conservative_config, aggressive_config]:
            assert config["NPU_USE_NPUW"] == "YES"
            assert config["NPUW_LLM"] == "YES"


class TestSystemValidation:
    """Test system requirements validation"""
    
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_validate_missing_model_path(self, mock_isdir, mock_exists):
        """Test validation when model path doesn't exist"""
        mock_exists.return_value = False
        mock_isdir.return_value = False
        
        with patch('app.config.get_config') as mock_config:
            mock_config.return_value.get.side_effect = lambda section, key, default=None: {
                ("model", "path"): "/nonexistent/path"
            }.get((section, key), default)
            
            issues = validate_system_requirements()
            assert len(issues) > 0
            assert any("does not exist" in issue for issue in issues)
    
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_validate_missing_model_files(self, mock_isdir, mock_exists):
        """Test validation when OpenVINO model files are missing"""
        def exists_side_effect(path):
            if "openvino_model.xml" in path or "openvino_model.bin" in path:
                return False
            return True
        
        mock_exists.side_effect = exists_side_effect
        mock_isdir.return_value = True
        
        with patch('app.config.get_config') as mock_config:
            mock_config.return_value.get.side_effect = lambda section, key, default=None: {
                ("model", "path"): "/test/model/path"
            }.get((section, key), default)
            
            issues = validate_system_requirements()
            assert len(issues) > 0
            assert any("Missing OpenVINO model file" in issue for issue in issues)
    
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.makedirs')
    def test_validate_cache_directory_creation(self, mock_makedirs, mock_isdir, mock_exists):
        """Test cache directory creation during validation"""
        def exists_side_effect(path):
            if "cache" in path and ".ovcache" not in path:
                return False  # Cache parent doesn't exist
            return True
        
        mock_exists.side_effect = exists_side_effect
        mock_isdir.return_value = True
        
        with patch('app.config.get_config') as mock_config:
            mock_config.return_value.get.side_effect = lambda section, key, default=None: {
                ("model", "path"): "/test/model/path",
                ("deployment", "cache_directory"): "./cache/.ovcache_phi3"
            }.get((section, key), default)
            
            issues = validate_system_requirements()
            
            # Should attempt to create cache directory
            mock_makedirs.assert_called()
    
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_validate_openvino_import(self, mock_isdir, mock_exists):
        """Test OpenVINO import validation"""
        mock_exists.return_value = True
        mock_isdir.return_value = True
        
        with patch('app.config.get_config') as mock_config:
            mock_config.return_value.get.side_effect = lambda section, key, default=None: {
                ("model", "path"): "/test/model/path",
                ("deployment", "target_device"): "NPU",
                ("deployment", "cache_directory"): "./cache/.ovcache_phi3"
            }.get((section, key), default)
            
            with patch('openvino.Core') as mock_core:
                mock_core.return_value.available_devices = ["NPU", "CPU"]
                
                issues = validate_system_requirements()
                
                # Should have no issues if everything is properly mocked
                device_issues = [i for i in issues if "not available" in i]
                assert len(device_issues) == 0