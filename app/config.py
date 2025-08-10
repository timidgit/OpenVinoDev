"""
Configuration Management Module
=============================

Handles configuration loading from multiple sources including JSON files,
environment variables, and command-line arguments.
"""

import json
import os
from typing import Any, Dict
from typing_extensions import TypedDict


# Type definitions
ConfigDict = Dict[str, Any]


class ConfigurationLoader:
    """Load and manage application configuration from multiple sources"""
    
    def __init__(self, config_file: str = "config.json") -> None:
        """
        Initialize configuration loader.
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config_file = config_file
        self._config: ConfigDict = {}
        self.load_configuration()
    
    def load_configuration(self) -> None:
        """Load configuration from file and environment variables"""
        # Load default configuration
        self._config = self._get_default_config()
        
        # Try to load from file
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                self._merge_config(file_config)
                print(f"✅ Loaded configuration from {self.config_file}")
            else:
                print(f"📝 Using default configuration (no {self.config_file} found)")
        except Exception as e:
            print(f"⚠️ Failed to load {self.config_file}: {e}")
            print("📝 Using default configuration")
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _get_default_config(self) -> ConfigDict:
        """Get default configuration values"""
        return {
            "model": {
                "path": "./models/qwen3-8b-int4-cw-ov",
                "name": "Qwen3-8B",
                "type": "qwen3"
            },
            "deployment": {
                "target_device": "NPU",
                "npu_profile": "balanced",
                "fallback_device": "CPU",
                "cache_directory": "./cache/.ovcache_qwen3"
            },
            "generation": {
                "max_new_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "repetition_penalty": 1.1,
                "do_sample": True
            },
            "ui": {
                "max_message_length": 400,
                "max_conversation_tokens": 1800,
                "emergency_limit": 2048,
                "show_performance_metrics": True,
                "theme": "soft"
            },
            "performance": {
                "generation_timeout": 30.0,
                "truncation_warning_delay": 0.5,
                "ui_update_interval": 0.1
            }
        }
    
    def _merge_config(self, new_config: ConfigDict) -> None:
        """Merge new configuration with existing configuration"""
        def merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> None:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self._config, new_config)
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides"""
        env_mappings = {
            "QWEN3_MODEL_PATH": ("model", "path"),  # Backward compatibility
            "MODEL_PATH": ("model", "path"),        # Generic name for any model
            "TARGET_DEVICE": ("deployment", "target_device"),
            "NPU_PROFILE": ("deployment", "npu_profile"),
            "CACHE_DIR": ("deployment", "cache_directory"),
            "MAX_MESSAGE_LENGTH": ("ui", "max_message_length"),
            "GENERATION_TIMEOUT": ("performance", "generation_timeout")
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion based on key
                if key in ["max_message_length"]:
                    value = int(value)
                elif key in ["generation_timeout"]:
                    value = float(value)
                elif key in ["show_performance_metrics", "do_sample"]:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                self._config[section][key] = value
                print(f"🔧 Environment override: {env_var} = {value}")
    
    def update_from_args(self, args) -> None:
        """
        Update configuration from command-line arguments.
        
        Args:
            args: Parsed arguments from argparse
        """
        if hasattr(args, 'model_path') and args.model_path:
            self._config['model']['path'] = args.model_path
            
        if hasattr(args, 'device') and args.device:
            self._config['deployment']['target_device'] = args.device
            
        if hasattr(args, 'npu_profile') and args.npu_profile:
            self._config['deployment']['npu_profile'] = args.npu_profile
            
        if hasattr(args, 'cache_dir') and args.cache_dir:
            self._config['deployment']['cache_directory'] = args.cache_dir
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(section, {}).get(key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self._config.get(section, {})
    
    @property
    def config(self) -> ConfigDict:
        """Get full configuration"""
        return self._config.copy()


# Global configuration instance - will be initialized by main.py
config: ConfigurationLoader = None


def initialize_config(config_file: str = "config.json", args=None) -> ConfigurationLoader:
    """
    Initialize global configuration instance.
    
    Args:
        config_file: Path to configuration file
        args: Command-line arguments from argparse
        
    Returns:
        Initialized ConfigurationLoader instance
    """
    global config
    config = ConfigurationLoader(config_file)
    
    if args:
        config.update_from_args(args)
    
    return config


def get_config() -> ConfigurationLoader:
    """Get the global configuration instance"""
    if config is None:
        raise RuntimeError("Configuration not initialized. Call initialize_config() first.")
    return config