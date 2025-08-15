"""
Enhanced OpenVINO GenAI Chat Application
========================================

A modular, production-ready implementation of Phi-3-mini-128k-instruct chat interface
using OpenVINO GenAI with Intel NPU optimization and RAG capabilities.

Copyright (c) 2025 sbran
Licensed under the MIT License - see LICENSE file for details

Modules:
--------
- config: Configuration management and environment handling
- model: Pipeline deployment and system initialization  
- streamer: Token streaming and filtering for LLMs
- chat: Core chat processing with RAG integration
- ui: Gradio interface creation and event handling
"""

import warnings
from typing import Any

__version__ = "2.0.0"
__author__ = "sbran"
__license__ = "MIT"

# Import key components for easy access
from .config import ConfigurationLoader
from .model import deploy_llm_pipeline, initialize_system_with_validation
from .streamer import EnhancedLLMStreamer
from .chat import enhanced_llm_chat
from .ui import create_enhanced_interface


def enhanced_qwen3_chat(*args, **kwargs) -> Any:
    """
    Deprecated wrapper for enhanced_llm_chat.
    
    This function maintains backward compatibility for legacy code that uses
    the 'qwen3' naming convention. It now processes Phi-3 models.
    
    Args:
        *args: Arguments passed to enhanced_llm_chat
        **kwargs: Keyword arguments passed to enhanced_llm_chat
        
    Returns:
        Result from enhanced_llm_chat
        
    Deprecated:
        Use enhanced_llm_chat instead. This wrapper will be removed in v3.0.0
    """
    warnings.warn(
        "enhanced_qwen3_chat is deprecated and will be removed in v3.0.0. "
        "Use enhanced_llm_chat instead. Note: This function now processes Phi-3 models, "
        "not Qwen3 models.",
        DeprecationWarning,
        stacklevel=2
    )
    return enhanced_llm_chat(*args, **kwargs)


__all__ = [
    "ConfigurationLoader",
    "deploy_llm_pipeline", 
    "initialize_system_with_validation",
    "EnhancedLLMStreamer",
    "enhanced_llm_chat",
    "create_enhanced_interface",
    # Deprecated functions (will be removed in v3.0.0)
    "enhanced_qwen3_chat"
]