"""
Enhanced Qwen3 OpenVINO GenAI Chat Application
==============================================

A modular, production-ready implementation of Qwen3-8B chat interface
using OpenVINO GenAI with Intel NPU optimization and RAG capabilities.

Copyright (c) 2025 sbran
Licensed under the MIT License - see LICENSE file for details

Modules:
--------
- config: Configuration management and environment handling
- model: Pipeline deployment and system initialization  
- streamer: Token streaming and filtering for Qwen3
- chat: Core chat processing with RAG integration
- ui: Gradio interface creation and event handling
"""

__version__ = "2.0.0"
__author__ = "sbran"
__license__ = "MIT"

# Import key components for easy access
from .config import ConfigurationLoader
from .model import deploy_qwen3_pipeline, initialize_system_with_validation
from .streamer import EnhancedQwen3Streamer
from .chat import enhanced_qwen3_chat
from .ui import create_enhanced_interface

__all__ = [
    "ConfigurationLoader",
    "deploy_qwen3_pipeline", 
    "initialize_system_with_validation",
    "EnhancedQwen3Streamer",
    "enhanced_qwen3_chat",
    "create_enhanced_interface"
]