#!/usr/bin/env python3
"""
Enhanced Qwen3 OpenVINO GenAI Chat Application

Copyright (c) 2025 sbran
Licensed under the MIT License - see LICENSE file for details

This application integrates with third-party components:
- OpenVINO GenAI (Apache 2.0, Intel Corporation)
- Qwen3 models (Apache 2.0 with additional terms, Alibaba Cloud)  
- Gradio (Apache 2.0, HuggingFace)
- Transformers (Apache 2.0, HuggingFace)

See ACKNOWLEDGMENTS.md for detailed attributions.

Features:
- Complete Qwen3-specific NPUW configuration and optimization
- Proper special token filtering for 26+ Qwen3 tokens
- Official Gradio streaming and ChatInterface patterns  
- Professional performance monitoring and diagnostics
- Robust error handling and deployment strategies
- Chat template support and session management

Based on enhanced context from:
- qwen3_model_context/ (model-specific optimizations)
- gradio_patterns/ (official Gradio best practices)  
- gradio_testing/ (professional testing patterns)

This is a hobby/educational project demonstrating OpenVINO GenAI capabilities.
"""

import gradio as gr
import openvino_genai as ov_genai
from transformers import AutoTokenizer

# Try to import OpenVINO properties with fallback for different versions
try:
    import openvino.properties as props
    import openvino.properties.hint as hints
    OPENVINO_PROPERTIES_AVAILABLE = True
    print("✅ OpenVINO properties imported successfully")
except ImportError as e:
    print(f"⚠️ OpenVINO properties not available: {e}")
    print("🔄 Using fallback configuration...")
    OPENVINO_PROPERTIES_AVAILABLE = False
    # Create mock objects for compatibility
    class MockHints:
        class PerformanceMode:
            LATENCY = "LATENCY"
            THROUGHPUT = "THROUGHPUT"
    
    class MockProps:
        class cache_dir:
            pass
        class streams:
            class num:
                pass
        class inference_num_threads:
            pass
    
    hints = MockHints()
    props = MockProps()
import time
import queue
import threading
import json
from typing import Optional, Dict, Any, List, Tuple, Iterator, Union, Callable
from typing_extensions import TypedDict, Literal
from dataclasses import dataclass, asdict
from pathlib import Path

# Type definitions for better code clarity
DeviceType = Literal["NPU", "CPU", "GPU", "AUTO"]
ProfileType = Literal["conservative", "balanced", "aggressive"]
ConfigDict = Dict[str, Any]
ChatMessage = TypedDict('ChatMessage', {'role': str, 'content': str})
ChatHistory = List[ChatMessage]

# Configuration result type
class DeploymentResult(TypedDict):
    pipeline: Any  # ov_genai.LLMPipeline
    device: str
    config: str
    load_time: float

# Import enhanced context patterns
import sys
import os
context_path = os.path.join(os.path.dirname(__file__), "context")
sys.path.insert(0, context_path)

# RAG system imports with fallback
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    RAG_AVAILABLE = True
    print("✅ RAG dependencies loaded successfully")
except ImportError as e:
    print(f"⚠️ RAG dependencies not available: {e}")
    print("📝 Install with: pip install langchain faiss-cpu sentence-transformers")
    RAG_AVAILABLE = False

# Import Qwen3-specific optimizations
try:
    from qwen3_model_context.npu_optimization import (
        Qwen3NPUConfigBuilder, 
        Qwen3NPUDeployment,
        Qwen3NPUPerformanceMonitor,
        QWEN3_NPU_PROFILES
    )
    from qwen3_model_context.special_tokens import (
        Qwen3StreamingFilter,
        Qwen3TokenFilter, 
        Qwen3ChatTemplate,
        QWEN3_SPECIAL_TOKENS
    )
    from qwen3_model_context.model_architecture import (
        QWEN3_8B_ARCHITECTURE,
        initialize_qwen3_pipeline
    )
    ENHANCED_CONTEXT_AVAILABLE = True
    print("✅ Enhanced Qwen3 context loaded successfully")
except ImportError as e:
    print(f"⚠️ Enhanced context not available: {e}")
    print("📝 Using fallback patterns - consider updating context path")
    ENHANCED_CONTEXT_AVAILABLE = False

# --- Configuration System ---
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
            "QWEN3_MODEL_PATH": ("model", "path"),
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

# Initialize configuration system
config = ConfigurationLoader()

# --- Constants and Configuration ---
# Use configuration system instead of hardcoded values
MODEL_PATH = config.get("model", "path", "./models/qwen3-8b-int4-cw-ov")
DEVICE = config.get("deployment", "target_device", "NPU")
CACHE_DIR = config.get("deployment", "cache_directory", "./cache/.ovcache_qwen3_enhanced")

# Qwen3-optimized settings from configuration
MAX_CONVERSATION_TOKENS = config.get("ui", "max_conversation_tokens", 1800)
EMERGENCY_LIMIT = config.get("ui", "emergency_limit", 2048)
MAX_MESSAGE_LENGTH = config.get("ui", "max_message_length", 400)
NPU_OPTIMIZATION_PROFILE = config.get("deployment", "npu_profile", "balanced")

@dataclass
class SystemMetrics:
    """Comprehensive system metrics tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    avg_first_token_latency: float = 0.0
    avg_tokens_per_second: float = 0.0
    total_tokens_generated: int = 0
    session_start_time: float = 0.0
    device_used: str = "unknown"
    config_used: str = "unknown"
    model_load_time: float = 0.0
    cache_hits: int = 0
    compilation_errors: int = 0
    special_tokens_filtered: int = 0
    
    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to user-friendly display format"""
        session_duration = time.time() - self.session_start_time
        success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100
        
        return {
            "Session Duration": f"{session_duration:.1f}s",
            "Total Requests": self.total_requests,
            "Success Rate": f"{success_rate:.1f}%",
            "Avg Response Time": f"{self.avg_response_time:.2f}s",
            "Avg Tokens/Second": f"{self.avg_tokens_per_second:.1f}",
            "Total Tokens Generated": self.total_tokens_generated,
            "Device": self.device_used,
            "Configuration": self.config_used,
            "Model Load Time": f"{self.model_load_time:.1f}s",
            "Special Tokens Filtered": self.special_tokens_filtered,
            "Cache Directory": CACHE_DIR
        }

# Global metrics instance
system_metrics = SystemMetrics(session_start_time=time.time())

# --- RAG System ---
class DocumentRAGSystem:
    """Retrieval-Augmented Generation system for document processing"""
    
    def __init__(self):
        """Initialize RAG system with fallback handling"""
        self.vector_store = None
        self.embeddings = None
        self.text_splitter = None
        self.processed_docs_count = 0
        self.available = RAG_AVAILABLE
        
        if RAG_AVAILABLE:
            try:
                # Initialize embeddings model (lightweight for fast loading)
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                
                # Initialize text splitter with optimized settings
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,  # Smaller chunks for better retrieval
                    chunk_overlap=100,  # Overlap for context preservation
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                print("✅ RAG system initialized successfully")
                
            except Exception as e:
                print(f"⚠️ RAG initialization failed: {e}")
                self.available = False
        else:
            print("📝 RAG system not available - install dependencies to enable")
    
    def process_uploaded_file(self, file_path: str, file_name: str) -> str:
        """
        Process uploaded file for RAG retrieval.
        
        Args:
            file_path: Path to the uploaded file
            file_name: Original name of the file
            
        Returns:
            Status message about processing result
        """
        if not self.available:
            return "❌ RAG system not available. Install langchain and faiss-cpu to enable document processing."
        
        try:
            # Read file content with encoding detection
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
            
            if not text.strip():
                return f"⚠️ File '{file_name}' appears to be empty."
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                return f"⚠️ No processable content found in '{file_name}'."
            
            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_texts(
                    texts=chunks, 
                    embedding=self.embeddings,
                    metadatas=[{"source": file_name, "chunk": i} for i in range(len(chunks))]
                )
            else:
                # Add new documents to existing store
                new_store = FAISS.from_texts(
                    texts=chunks, 
                    embedding=self.embeddings,
                    metadatas=[{"source": file_name, "chunk": i} for i in range(len(chunks))]
                )
                self.vector_store.merge_from(new_store)
            
            self.processed_docs_count += 1
            
            return f"✅ Successfully processed '{file_name}': {len(chunks)} chunks created from {len(text):,} characters. Ready to answer questions about this document."
            
        except Exception as e:
            return f"❌ Error processing '{file_name}': {str(e)}"
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User question to find relevant context for
            k: Number of top chunks to retrieve
            
        Returns:
            Concatenated context from relevant document chunks
        """
        if not self.available or self.vector_store is None:
            return ""
        
        try:
            # Search for relevant documents
            docs = self.vector_store.similarity_search(query, k=k)
            
            if not docs:
                return ""
            
            # Format context with source attribution
            context_parts = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                content = doc.page_content.strip()
                context_parts.append(f"[From {source}]\n{content}")
            
            return "\n\n---\n\n".join(context_parts)
            
        except Exception as e:
            print(f"⚠️ Context retrieval error: {e}")
            return ""
    
    def clear_documents(self) -> str:
        """Clear all processed documents"""
        self.vector_store = None
        self.processed_docs_count = 0
        return "✅ All documents cleared from memory."
    
    def get_status(self) -> dict:
        """Get current RAG system status"""
        return {
            "Available": self.available,
            "Documents Processed": self.processed_docs_count,
            "Vector Store": "Loaded" if self.vector_store is not None else "Empty",
            "Embedding Model": "all-MiniLM-L6-v2" if self.available else "None"
        }

# Global RAG system instance
rag_system = DocumentRAGSystem()

# Enhanced system prompt with Qwen3 optimization
DEFAULT_SYSTEM_PROMPT = """You are a helpful, concise AI assistant powered by Qwen3-8B running on Intel NPU via OpenVINO GenAI. 

Key behaviors:
- Provide accurate, well-structured responses
- Be concise but comprehensive 
- Use clear formatting when helpful
- Acknowledge when you're uncertain
- Optimize for NPU constraints (prefer shorter, focused responses)

You excel at: reasoning, coding, analysis, creative writing, and technical explanations."""

# Current system prompt (can be modified by user)
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

# --- Enhanced Configuration Management ---
class Qwen3ConfigurationManager:
    """Advanced configuration management with Qwen3 optimization"""
    
    def __init__(self, profile: ProfileType = "balanced") -> None:
        """
        Initialize configuration manager with specified profile.
        
        Args:
            profile: NPU optimization profile (conservative, balanced, aggressive)
        """
        self.profile = profile
        self.config_builder: Optional[Any] = None  # Qwen3NPUConfigBuilder if available
        self.performance_monitor: Optional[Any] = None  # Qwen3NPUPerformanceMonitor if available
        
        if ENHANCED_CONTEXT_AVAILABLE:
            self.config_builder = Qwen3NPUConfigBuilder(profile)
            self.performance_monitor = Qwen3NPUPerformanceMonitor()
    
    def get_npu_config(self) -> ConfigDict:
        """
        Get complete NPU configuration with NPUW optimization.
        
        Returns:
            Dictionary containing NPU-specific configuration parameters
        """
        if ENHANCED_CONTEXT_AVAILABLE and self.config_builder:
            # Use enhanced Qwen3-specific configuration
            return self.config_builder.build_complete_config()
        else:
            # Fallback configuration with compatibility handling
            config = {
                "NPU_USE_NPUW": "YES",
                "NPUW_LLM": "YES", 
                "NPUW_LLM_BATCH_DIM": 0,
                "NPUW_LLM_SEQ_LEN_DIM": 1,
                "NPUW_LLM_MAX_PROMPT_LEN": 2048,
                "NPUW_LLM_MIN_RESPONSE_LEN": 256,
                "CACHE_MODE": "OPTIMIZE_SPEED",
                "NPUW_LLM_PREFILL_HINT": "BEST_PERF",
                "NPUW_LLM_GENERATE_HINT": "BEST_PERF"
            }
            
            # Add OpenVINO properties if available (no generic PERFORMANCE_HINT for NPU)
            if OPENVINO_PROPERTIES_AVAILABLE:
                config.update({
                    props.cache_dir: CACHE_DIR
                })
            else:
                config.update({
                    "CACHE_DIR": CACHE_DIR
                })
            
            return config
    
    def get_cpu_config(self) -> ConfigDict:
        """
        Get optimized CPU configuration.
        
        Returns:
            Dictionary containing CPU-specific configuration parameters
        """
        if ENHANCED_CONTEXT_AVAILABLE and self.config_builder:
            return self.config_builder.build_complete_config()
        else:
            config = {
                "MAX_PROMPT_LEN": 4096,  # Larger context on CPU
                "MIN_RESPONSE_LEN": 512
            }
            
            # Add OpenVINO properties if available
            if OPENVINO_PROPERTIES_AVAILABLE:
                config.update({
                    hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                    props.cache_dir: CACHE_DIR + "_cpu",
                    props.streams.num: 2,
                    props.inference_num_threads: 0  # Auto-detect
                })
            else:
                config.update({
                    "PERFORMANCE_HINT": "THROUGHPUT",
                    "CACHE_DIR": CACHE_DIR + "_cpu",
                    "NUM_STREAMS": 2,
                    "INFERENCE_NUM_THREADS": 0  # Auto-detect
                })
            
            return config

# --- Enhanced Pipeline Deployment ---
def deploy_qwen3_pipeline(
    model_path: str, 
    target_device: DeviceType, 
    profile: ProfileType = "balanced"
) -> Tuple[Any, str, str, float]:
    """
    Deploy Qwen3 pipeline with comprehensive error handling and optimization.
    
    Args:
        model_path: Path to the Qwen3 OpenVINO model directory
        target_device: Target device for deployment (NPU, CPU, GPU, AUTO)
        profile: NPU optimization profile
        
    Returns:
        Tuple of (pipeline, device_used, config_used, load_time)
        
    Raises:
        RuntimeError: If all deployment configurations fail
    """
    load_start_time = time.time()
    
    if ENHANCED_CONTEXT_AVAILABLE:
        print(f"🚀 Deploying Qwen3 with enhanced context (profile: {profile})")
        
        # Use enhanced deployment
        deployment = Qwen3NPUDeployment(model_path, profile)
        pipeline = deployment.deploy()
        
        if pipeline:
            load_time = time.time() - load_start_time
            return pipeline, target_device, f"enhanced_{profile}", load_time
        else:
            print("⚠️ Enhanced deployment failed, falling back to manual configuration")
    
    # Fallback to manual configuration
    print(f"🔄 Using manual pipeline deployment (target: {target_device})")
    
    config_manager = Qwen3ConfigurationManager(profile)
    
    configurations = []
    
    # Create basic configurations with compatibility handling
    if OPENVINO_PROPERTIES_AVAILABLE:
        basic_npu_config = {hints.performance_mode: hints.PerformanceMode.LATENCY, props.cache_dir: CACHE_DIR}
        basic_cpu_config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT, props.cache_dir: CACHE_DIR}
    else:
        basic_npu_config = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": CACHE_DIR}
        basic_cpu_config = {"PERFORMANCE_HINT": "THROUGHPUT", "CACHE_DIR": CACHE_DIR}
    
    if target_device == "NPU":
        configurations = [
            ("enhanced_npu_qwen3", target_device, config_manager.get_npu_config()),
            ("basic_npu", target_device, basic_npu_config),
            ("minimal_npu", target_device, {}),
            ("cpu_fallback", "CPU", config_manager.get_cpu_config())
        ]
    else:
        configurations = [
            ("optimized_cpu_qwen3", target_device, config_manager.get_cpu_config()),
            ("basic_cpu", target_device, basic_cpu_config),
            ("minimal_cpu", target_device, {})
        ]
    
    for config_name, device, config in configurations:
        try:
            print(f"🔄 Trying {device} with {config_name} configuration...")
            
            if ENHANCED_CONTEXT_AVAILABLE:
                # Use enhanced initialization if available
                pipeline = initialize_qwen3_pipeline(model_path, device, **config)
            else:
                # Fallback initialization
                if config:
                    pipeline = ov_genai.LLMPipeline(model_path, device, **config)
                else:
                    pipeline = ov_genai.LLMPipeline(model_path, device)
                
            load_time = time.time() - load_start_time
            print(f"✅ Success: {device} with {config_name} ({load_time:.1f}s)")
            return pipeline, device, config_name, load_time
            
        except Exception as e:
            print(f"⚠️ {config_name} failed: {e}")
            if "compile" in str(e).lower():
                system_metrics.compilation_errors += 1
            continue
    
    raise RuntimeError("All configurations failed. Check model path, device drivers, and NPUW configuration.")

# --- Enhanced Streaming with Qwen3 Token Filtering ---
class EnhancedQwen3Streamer(ov_genai.StreamerBase):
    """
    Production-ready streamer with Qwen3-specific optimizations:
    - Proper special token filtering (26+ tokens)
    - Performance monitoring
    - Robust error handling
    - Token-level streaming control
    """
    
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_queue = queue.Queue()
        self.accumulated_tokens = []
        self.current_text = ""
        self.start_time = time.time()
        self.first_token_time = None
        self.tokens_generated = 0
        
        # Initialize Qwen3-specific filtering
        if ENHANCED_CONTEXT_AVAILABLE:
            self.token_filter = Qwen3StreamingFilter()
            print("✅ Using enhanced Qwen3 token filtering")
        else:
            self.token_filter = None
            print("⚠️ Using basic token filtering")
    
    def put(self, token_id: int) -> bool:
        """Process token with Qwen3-specific filtering"""
        self.accumulated_tokens.append(token_id)
        self.tokens_generated += 1
        
        # Record first token latency
        if self.first_token_time is None:
            self.first_token_time = time.time()
        
        try:
            # Decode with special token handling
            if ENHANCED_CONTEXT_AVAILABLE and self.token_filter:
                # Use enhanced filtering
                try:
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                except:
                    token_text = f"[UNK_{token_id}]"
                
                # Process through Qwen3 filter
                display_text = self.token_filter.process_token(token_id, token_text)
                
                if display_text is not None:
                    self.current_text += display_text
                    self.text_queue.put(display_text)
                else:
                    # Token was filtered (special token)
                    system_metrics.special_tokens_filtered += 1
                    
            else:
                # Fallback filtering
                try:
                    # Decode incrementally
                    full_text = self.tokenizer.decode(self.accumulated_tokens, skip_special_tokens=True)
                    
                    if len(full_text) > len(self.current_text):
                        new_text = full_text[len(self.current_text):]
                        self.current_text = full_text
                        
                        # Basic special token filtering
                        if not self._is_special_token_text(new_text):
                            self.text_queue.put(new_text)
                except Exception as e:
                    print(f"⚠️ Decoding error: {e}")
                    return False
                    
        except Exception as e:
            print(f"❌ Token processing error: {e}")
            return False
        
        return False  # Continue generation
    
    def _is_special_token_text(self, text: str) -> bool:
        """Basic special token detection for fallback"""
        special_patterns = [
            '<|im_start|>', '<|im_end|>', '<|endoftext|>',
            '<think>', '</think>', '<tool_call>', '</tool_call>',
            '<|system|>', '<|user|>', '<|assistant|>'
        ]
        
        for pattern in special_patterns:
            if pattern in text:
                system_metrics.special_tokens_filtered += 1
                return True
        
        return False
    
    def end(self):
        """Finalize streaming and calculate performance metrics"""
        # Calculate performance metrics
        total_time = time.time() - self.start_time
        first_token_latency = (self.first_token_time - self.start_time) if self.first_token_time else 0
        tokens_per_second = self.tokens_generated / total_time if total_time > 0 else 0
        
        # Update global metrics
        system_metrics.avg_first_token_latency = (
            (system_metrics.avg_first_token_latency * (system_metrics.successful_requests - 1) + first_token_latency)
            / system_metrics.successful_requests
        ) if system_metrics.successful_requests > 0 else first_token_latency
        
        system_metrics.avg_tokens_per_second = (
            (system_metrics.avg_tokens_per_second * (system_metrics.successful_requests - 1) + tokens_per_second)
            / system_metrics.successful_requests
        ) if system_metrics.successful_requests > 0 else tokens_per_second
        
        system_metrics.total_tokens_generated += self.tokens_generated
        
        # Log performance
        print(f"🚀 Generation complete: {self.tokens_generated} tokens in {total_time:.2f}s")
        print(f"   First token: {first_token_latency:.3f}s, Speed: {tokens_per_second:.1f} tok/s")
        
        if ENHANCED_CONTEXT_AVAILABLE and self.token_filter:
            thinking_content = self.token_filter.get_thinking_content()
            if thinking_content.strip():
                print(f"🧠 Model thinking: {thinking_content[:100]}...")
        
        # Signal end of generation
        self.text_queue.put(None)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        item = self.text_queue.get()
        if item is None:
            raise StopIteration
        return item

# --- Enhanced Generation Configuration ---
# --- Security and Validation ---
class InputValidator:
    """Security-focused input validation and sanitization"""
    
    @staticmethod
    def validate_message(message: str) -> Tuple[bool, str]:
        """
        Validate user message for security and content policy compliance.
        
        Args:
            message: User input to validate
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if not message or not isinstance(message, str):
            return False, "Empty or invalid message"
        
        # Check for excessively long messages (security)
        if len(message) > 10000:  # Much higher than UI limit
            return False, "Message exceeds maximum length"
        
        # Check for potential injection patterns
        suspicious_patterns = [
            r'<script[^>]*>',  # Script injection
            r'javascript:',     # JavaScript URLs
            r'data:.*base64',   # Data URLs
            r'eval\s*\(',      # Eval calls
            r'exec\s*\(',      # Exec calls
        ]
        
        import re
        for pattern in suspicious_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return False, "Message contains potentially unsafe content"
        
        # Check for excessive special characters (potential encoding attacks)
        special_char_ratio = len([c for c in message if not c.isalnum() and not c.isspace()]) / len(message)
        if special_char_ratio > 0.5:  # More than 50% special characters
            return False, "Message contains excessive special characters"
        
        return True, ""
    
    @staticmethod
    def sanitize_message(message: str) -> str:
        """
        Sanitize user message while preserving readability.
        
        Args:
            message: Raw user input
            
        Returns:
            Sanitized message safe for processing
        """
        # Remove null bytes and control characters (except newlines and tabs)
        sanitized = ''.join(char for char in message if ord(char) >= 32 or char in '\n\t')
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Limit consecutive repeated characters (potential DoS protection)
        import re
        sanitized = re.sub(r'(.)\1{10,}', r'\1\1\1', sanitized)
        
        return sanitized.strip()

def create_qwen3_generation_config() -> ov_genai.GenerationConfig:
    """
    Create optimized generation configuration for Qwen3 from configuration file.
    
    Returns:
        Configured GenerationConfig with security-conscious defaults
    """
    gen_config = ov_genai.GenerationConfig()
    
    # Load generation settings from configuration
    gen_settings = config.get_section("generation")
    
    if ENHANCED_CONTEXT_AVAILABLE:
        # Use Qwen3-specific defaults from context
        gen_config.do_sample = gen_settings.get("do_sample", True)
        gen_config.temperature = min(gen_settings.get("temperature", 0.6), 2.0)  # Security: cap temperature
        gen_config.top_p = min(gen_settings.get("top_p", 0.95), 1.0)  # Security: cap top_p
        gen_config.top_k = min(gen_settings.get("top_k", 20), 100)  # Security: reasonable top_k
        gen_config.max_new_tokens = min(gen_settings.get("max_new_tokens", 1024), 2048)  # Security: limit tokens
        gen_config.repetition_penalty = max(1.0, min(gen_settings.get("repetition_penalty", 1.1), 2.0))  # Security: reasonable range
    else:
        # Fallback configuration with security limits
        gen_config.do_sample = gen_settings.get("do_sample", True)
        gen_config.temperature = min(gen_settings.get("temperature", 0.7), 2.0)
        gen_config.top_p = min(gen_settings.get("top_p", 0.9), 1.0)
        gen_config.top_k = min(gen_settings.get("top_k", 50), 100)
        gen_config.max_new_tokens = min(gen_settings.get("max_new_tokens", 1024), 2048)
        gen_config.repetition_penalty = max(1.0, min(gen_settings.get("repetition_penalty", 1.1), 2.0))
    
    return gen_config

# --- Smart Message Processing ---
def process_user_message(message: str, history: ChatHistory) -> Tuple[str, bool]:
    """
    Process user message with Qwen3-optimized handling.
    
    Args:
        message: Raw user input message
        history: Current chat conversation history
        
    Returns:
        Tuple of (processed_message, was_truncated)
    """
    original_length = len(message)
    
    # Handle overly long messages
    if original_length > MAX_MESSAGE_LENGTH:
        # Smart truncation
        if '.' in message:
            sentences = message.split('.')
            truncated = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) + 1 <= MAX_MESSAGE_LENGTH * 0.85:
                    truncated.append(sentence)
                    current_length += len(sentence) + 1
                else:
                    break
            
            if truncated:
                processed = '. '.join(truncated) + '.'
                if len(processed) < original_length * 0.5:
                    processed = message[:MAX_MESSAGE_LENGTH-50] + "..."
            else:
                processed = message[:MAX_MESSAGE_LENGTH-50] + "..."
        else:
            processed = message[:MAX_MESSAGE_LENGTH-50] + "..."
        
        print(f"📏 Message truncated: {original_length} → {len(processed)} chars")
        return processed, True
    
    return message, False

# --- Core Chat Processing Functions ---
def prepare_chat_input(message: str, history: ChatHistory) -> Tuple[str, bool, ChatHistory]:
    """
    Prepare and validate chat input with smart message handling and security validation.
    
    Args:
        message: Raw user input
        history: Current chat history
        
    Returns:
        Tuple of (processed_message, was_truncated, updated_history)
        
    Raises:
        ValueError: If message fails security validation
    """
    # Input validation
    if not message.strip():
        return message, False, history
    
    # Security validation
    is_valid, reason = InputValidator.validate_message(message)
    if not is_valid:
        error_history = history.copy()
        error_history.append({
            "role": "assistant", 
            "content": f"🚫 Message rejected: {reason}. Please try a different message."
        })
        raise ValueError(f"Security validation failed: {reason}")
    
    # Sanitize input
    sanitized_message = InputValidator.sanitize_message(message)
    
    # Process message with smart handling
    processed_message, was_truncated = process_user_message(sanitized_message, history)
    
    # Update history with user message and truncation warning if needed
    updated_history = history.copy()
    
    if was_truncated:
        truncation_warning = {
            "role": "assistant",
            "content": f"⚠️ Your message was truncated from {len(message):,} to {len(processed_message)} characters due to NPU memory limits. Processing the truncated version..."
        }
        updated_history.append({"role": "user", "content": message})
        updated_history.append(truncation_warning)
    else:
        updated_history.append({"role": "user", "content": processed_message})
    
    # Add assistant placeholder
    updated_history.append({"role": "assistant", "content": ""})
    
    return processed_message, was_truncated, updated_history

def execute_generation(processed_message: str, streamer: EnhancedQwen3Streamer) -> bool:
    """
    Execute model generation in a controlled manner.
    
    Args:
        processed_message: Message to generate response for
        streamer: Configured streamer for token processing
        
    Returns:
        True if generation succeeded, False otherwise
    """
    try:
        generation_config = create_qwen3_generation_config()
        pipe.generate(processed_message, generation_config, streamer)
        return True
    except Exception as e:
        print(f"❌ Generation error: {e}")
        # Send error through streamer
        error_msg = f"❌ Generation error: {str(e)[:100]}..."
        streamer.text_queue.put(error_msg)
        streamer.text_queue.put(None)
        return False

def stream_response_to_history(streamer: 'EnhancedQwen3Streamer', history: ChatHistory) -> Iterator[ChatHistory]:
    """
    Stream model response tokens to chat history.
    
    Args:
        streamer: Active streamer with generation in progress
        history: Chat history to update
        
    Yields:
        Updated history with streaming response
    """
    try:
        for chunk in streamer:
            if chunk:  # Only add non-empty chunks
                history[-1]["content"] += chunk
                yield history
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        history[-1]["content"] = f"❌ Streaming error: {str(e)[:100]}..."
        yield history

def handle_chat_error(error: Exception, history: ChatHistory) -> ChatHistory:
    """
    Handle chat errors with user-friendly messages.
    
    Args:
        error: Exception that occurred
        history: Current chat history
        
    Returns:
        Updated history with error message
    """
    print(f"❌ Chat function error: {error}")
    
    # Determine error type and provide helpful message
    error_message = "❌ An error occurred. "
    error_str = str(error).lower()
    
    if "memory" in error_str:
        error_message += "Memory limit reached. Try starting a new conversation."
    elif "token" in error_str or "length" in error_str:
        error_message += "Message too long. Please try a shorter message."
    elif "compile" in error_str:
        error_message += "NPU compilation issue. Check NPUW configuration."
    elif "timeout" in error_str:
        error_message += "Generation timed out. Try a simpler request."
    elif "device" in error_str:
        error_message += "Device error. NPU may not be available."
    else:
        error_message += f"Details: {str(error)[:100]}..."
    
    # Add error to history
    updated_history = history.copy()
    if not updated_history or updated_history[-1]["role"] != "assistant":
        updated_history.append({"role": "assistant", "content": error_message})
    else:
        updated_history[-1]["content"] = error_message
    
    return updated_history

# --- Enhanced Chat Function (Refactored) ---
def enhanced_qwen3_chat(message: str, history: ChatHistory) -> Iterator[ChatHistory]:
    """
    Enhanced chat function with comprehensive Qwen3 optimization and RAG support.
    
    This is the main chat processing function that handles user input,
    processes it through the Qwen3 model with optional document context,
    and streams back the response with comprehensive error handling and performance monitoring.
    
    Args:
        message: User input message to process
        history: Current chat conversation history
        
    Yields:
        Updated chat history with streaming response as it's generated
        
    Note:
        Uses global system_metrics for performance tracking and
        requires global pipe and tokenizer to be initialized.
    """
    global system_metrics
    
    request_start_time = time.time()
    system_metrics.total_requests += 1
    
    try:
        # Step 1: Prepare and validate input
        processed_message, was_truncated, updated_history = prepare_chat_input(message, history)
        
        # Early return for empty messages
        if not processed_message.strip():
            yield updated_history
            return
        
        # Show truncation warning with brief pause
        if was_truncated:
            yield updated_history
            time.sleep(0.5)  # Brief pause for user to see warning
        
        # Step 1.5: RAG Context Retrieval
        rag_context = ""
        if rag_system.available and rag_system.vector_store is not None:
            rag_context = rag_system.retrieve_context(processed_message)
            if rag_context:
                # Augment the message with context
                augmented_message = f"""Based on the following context from uploaded documents, please answer the user's question. If the context doesn't contain relevant information, please indicate that and provide a general response.

Context:
{rag_context}

Question: {processed_message}"""
                processed_message = augmented_message
                print(f"📚 Using RAG context: {len(rag_context)} characters from documents")
        
        # Step 2: Initialize streaming components
        streamer = EnhancedQwen3Streamer(tokenizer)
        
        # Step 3: Execute generation in separate thread
        def generation_worker():
            success = execute_generation(processed_message, streamer)
            if success:
                system_metrics.successful_requests += 1
            else:
                system_metrics.failed_requests += 1
        
        generation_thread = threading.Thread(target=generation_worker, daemon=True)
        generation_thread.start()
        
        # Step 4: Stream response to UI
        yield from stream_response_to_history(streamer, updated_history)
        
        # Step 5: Wait for generation completion with timeout
        generation_thread.join(timeout=30.0)
        if generation_thread.is_alive():
            print("⚠️ Generation timeout - thread still running")
        
        # Step 6: Update performance metrics
        elapsed_time = time.time() - request_start_time
        system_metrics.avg_response_time = (
            (system_metrics.avg_response_time * (system_metrics.total_requests - 1) + elapsed_time)
            / system_metrics.total_requests
        )
        
        print(f"📊 Request complete: {elapsed_time:.2f}s total")
        
    except Exception as e:
        system_metrics.failed_requests += 1
        error_history = handle_chat_error(e, history)
        yield error_history

# --- System Initialization ---
print("🚀 Initializing Enhanced Qwen3 Chat System")
print(f"📂 Model: {MODEL_PATH}")
print(f"🎯 Target Device: {DEVICE}")
print(f"📊 Optimization Profile: {NPU_OPTIMIZATION_PROFILE}")
print(f"🔧 Enhanced Context: {'Available' if ENHANCED_CONTEXT_AVAILABLE else 'Fallback Mode'}")

def validate_system_requirements() -> List[str]:
    """Validate system requirements and return list of issues."""
    issues = []
    
    # Check model path
    if not os.path.exists(MODEL_PATH):
        issues.append(f"Model path does not exist: {MODEL_PATH}")
    elif not os.path.isdir(MODEL_PATH):
        issues.append(f"Model path is not a directory: {MODEL_PATH}")
    else:
        # Check for required OpenVINO files
        required_files = ['openvino_model.xml', 'openvino_model.bin']
        for file_name in required_files:
            if not os.path.exists(os.path.join(MODEL_PATH, file_name)):
                issues.append(f"Missing OpenVINO model file: {file_name}")
    
    # Check cache directory
    cache_dir = os.path.dirname(CACHE_DIR)
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except PermissionError:
            issues.append(f"Cannot create cache directory: {cache_dir} (permission denied)")
        except Exception as e:
            issues.append(f"Cannot create cache directory: {cache_dir} ({str(e)})")
    
    # Check OpenVINO installation
    try:
        import openvino as ov
        core = ov.Core()
        available_devices = core.available_devices
        if DEVICE not in available_devices and DEVICE != "AUTO":
            issues.append(f"Target device '{DEVICE}' not available. Available: {available_devices}")
    except Exception as e:
        issues.append(f"OpenVINO not properly installed: {str(e)}")
    
    return issues

def initialize_system_with_validation():
    """Initialize system with comprehensive validation and error handling."""
    global pipe, tokenizer, system_metrics
    
    print("🔍 Validating system requirements...")
    issues = validate_system_requirements()
    
    if issues:
        print("❌ System validation failed:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\n🔧 Suggested fixes:")
        print("   • Set QWEN3_MODEL_PATH environment variable to correct model location")
        print("   • Install OpenVINO with: pip install openvino")
        print("   • For NPU: Install Intel NPU drivers from official site")
        print("   • Ensure model is in OpenVINO format (.xml/.bin files)")
        raise SystemExit(1)
    
    try:
        print("🚀 Initializing Enhanced Qwen3 Chat System...")
        
        # Deploy pipeline with comprehensive error handling
        pipe, device_used, config_used, load_time = deploy_qwen3_pipeline(
            MODEL_PATH, DEVICE, NPU_OPTIMIZATION_PROFILE
        )
        
        # Update system metrics
        system_metrics.device_used = device_used
        system_metrics.config_used = config_used
        system_metrics.model_load_time = load_time
        
        # Initialize tokenizer with error handling
        print("📚 Loading Qwen3 tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            
            # Configure tokenizer for Qwen3
            if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                
        except Exception as tokenizer_error:
            print(f"⚠️ Tokenizer loading failed: {tokenizer_error}")
            print("🔄 Attempting fallback tokenizer initialization...")
            try:
                # Fallback: try without trust_remote_code
                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
                if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                print("✅ Fallback tokenizer loaded successfully")
            except Exception as fallback_error:
                print(f"❌ Fallback tokenizer also failed: {fallback_error}")
                raise RuntimeError("Unable to initialize tokenizer with any method") from fallback_error
        
        print(f"✅ System Ready!")
        print(f"   Device: {device_used}")
        print(f"   Config: {config_used}")
        print(f"   Load Time: {load_time:.1f}s")
        print(f"   Model Path: {MODEL_PATH}")
        print(f"   Tokenizer: {tokenizer.__class__.__name__}")
        if ENHANCED_CONTEXT_AVAILABLE:
            print(f"   Special Tokens: {len(QWEN3_SPECIAL_TOKENS)} Qwen3 tokens loaded")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        print("\n🔧 Detailed diagnostics:")
        print(f"   Model Path: {MODEL_PATH}")
        print(f"   Target Device: {DEVICE}")
        print(f"   Cache Directory: {CACHE_DIR}")
        print(f"   Enhanced Context: {ENHANCED_CONTEXT_AVAILABLE}")
        
        # Provide specific guidance based on error type
        error_str = str(e).lower()
        if "compile" in error_str:
            print("\n💡 NPU Compilation Error - Try:")
            print("   • Verify NPU drivers are installed")
            print("   • Check NPUW configuration compatibility")
            print("   • Try CPU fallback with: export TARGET_DEVICE=CPU")
        elif "file" in error_str or "path" in error_str:
            print("\n💡 File/Path Error - Try:")
            print("   • Verify model path contains .xml and .bin files")
            print("   • Check file permissions and access rights")
        elif "memory" in error_str:
            print("\n💡 Memory Error - Try:")
            print("   • Use conservative NPU profile")
            print("   • Ensure sufficient system RAM")
            print("   • Close other applications")
        
        raise SystemExit(1)

# Initialize system with enhanced error handling
try:
    initialize_system_with_validation()
except SystemExit:
    raise
except Exception as unexpected_error:
    print(f"💥 Unexpected initialization error: {unexpected_error}")
    print("🆘 This may be a bug - please report with full error details")
    raise

# --- Enhanced Gradio Interface ---
def create_enhanced_interface():
    """Create production-ready Gradio interface with advanced features"""
    
    # Custom CSS for professional appearance
    custom_css = """
    .gradio-container { max-width: 1400px; margin: auto; }
    .chatbot { height: 650px; }
    .metrics-panel { 
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #dee2e6;
    }
    .system-info {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 6px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    .warning-banner {
        background: #fff3cd;
        padding: 8px;
        border-radius: 4px;
        border-left: 4px solid #ffc107;
        margin: 5px 0;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate"
        ),
        title="Enhanced Qwen3 Chat",
        css=custom_css,
    ) as demo:
        
        # Header with system status
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(f"""
                # 🤖 Enhanced Qwen3 Chat System
                
                **Production-Ready Implementation with Complete Optimization**
                """)
            
            with gr.Column(scale=2, elem_classes=["system-info"]):
                system_status = gr.Markdown(f"""
                **Device**: {device_used} | **Config**: {config_used}  
                **Model**: Qwen3-8B INT4 | **Load Time**: {load_time:.1f}s  
                **Enhanced Context**: {'✅ Active' if ENHANCED_CONTEXT_AVAILABLE else '⚠️ Fallback'}  
                **Profile**: {NPU_OPTIMIZATION_PROFILE.title()}
                """)
        
        # Warning banner if fallback mode
        if not ENHANCED_CONTEXT_AVAILABLE:
            gr.Markdown("""
            <div class="warning-banner">
            ⚠️ <strong>Fallback Mode</strong>: Enhanced context not loaded. Some optimizations may be limited.
            </div>
            """)
        
        # Main chat interface using official ChatInterface pattern
        chatbot = gr.Chatbot(
            label=f"Conversation (Qwen3-8B on {device_used})",
            height=650,
            type='messages',
            avatar_images=(None, "🤖"),
            show_copy_button=True,
            show_share_button=False,
            bubble_full_width=False,
            render_markdown=True
        )
        
        # System prompt control
        with gr.Accordion("🎯 System Prompt Configuration", open=False):
            system_prompt_input = gr.Textbox(
                value=SYSTEM_PROMPT,
                lines=6,
                label="System Prompt",
                placeholder="Configure the AI's behavior and persona...",
                interactive=True,
                info="This prompt sets the AI's behavior, expertise, and response style. Changes take effect after clearing the chat."
            )
            
            with gr.Row():
                reset_prompt_btn = gr.Button("🔄 Reset to Default", size="sm")
                apply_prompt_btn = gr.Button("✅ Apply & Clear Chat", variant="primary", size="sm")
        
        # Document upload for RAG
        with gr.Accordion("📚 Document Upload (RAG)", open=False):
            with gr.Row():
                with gr.Column(scale=3):
                    file_upload = gr.File(
                        label="Upload Documents",
                        file_types=[".txt", ".md", ".py", ".js", ".html", ".css", ".json"],
                        file_count="multiple",
                        interactive=True
                    )
                    
                with gr.Column(scale=2):
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        interactive=False,
                        placeholder="No documents uploaded"
                    )
            
            with gr.Row():
                clear_docs_btn = gr.Button("🗑️ Clear Documents", variant="secondary", size="sm")
                rag_status_btn = gr.Button("📊 RAG Status", size="sm")
            
            if not RAG_AVAILABLE:
                gr.Markdown("""
                ⚠️ **RAG not available**: Install dependencies with:
                ```
                pip install langchain faiss-cpu sentence-transformers
                ```
                """)

        # Input controls
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder=f"Chat with Qwen3 on {device_used} (max {MAX_MESSAGE_LENGTH} chars)...",
                scale=7,
                max_lines=4,
                show_label=False,
                container=False
            )
            
            with gr.Column(scale=1):
                send_btn = gr.Button("💬 Send", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
        
        # Advanced controls panel
        with gr.Row():
            with gr.Column(scale=2):
                metrics_btn = gr.Button("📊 Performance Metrics", variant="secondary")
                system_btn = gr.Button("ℹ️ System Info", variant="secondary")
            
            with gr.Column(scale=2):
                if ENHANCED_CONTEXT_AVAILABLE:
                    profile_selector = gr.Dropdown(
                        choices=list(QWEN3_NPU_PROFILES.keys()),
                        value=NPU_OPTIMIZATION_PROFILE,
                        label="NPU Profile",
                        interactive=False  # Would need restart to change
                    )
                
                reset_metrics_btn = gr.Button("🔄 Reset Metrics", variant="secondary")
        
        # Collapsible metrics panel
        with gr.Row(visible=False) as metrics_row:
            with gr.Column(elem_classes=["metrics-panel"]):
                gr.Markdown("### 📊 Real-time Performance Metrics")
                metrics_json = gr.JSON(label="System Metrics", container=True)
                
                if ENHANCED_CONTEXT_AVAILABLE:
                    gr.Markdown("### 🎯 Qwen3-Specific Stats")
                    qwen3_stats = gr.JSON(label="Token Filtering & Processing", container=True)
        
        # Examples section
        with gr.Row():
            gr.Examples(
                examples=[
                    "Explain quantum computing in simple terms",
                    "Write a Python function to implement quicksort",
                    "What are the advantages of using Intel NPU for AI inference?",
                    "Compare different neural network architectures",
                    "Help me debug this code: def factorial(n): return n * factorial(n)",
                    "Explain the concept of attention in transformer models",
                    "What does the uploaded document say about...?",
                    "Summarize the key points from the uploaded files"
                ],
                inputs=msg_input,
                label="💡 Example Questions (Upload documents for context-aware answers)"
            )
        
        # Event handlers with enhanced functionality
        def handle_send(message, history):
            """Handle send with proper session management"""
            return enhanced_qwen3_chat(message, history)
        
        def handle_clear(current_system_prompt):
            """Handle clear with proper session reset"""
            global SYSTEM_PROMPT
            try:
                # Update global system prompt if changed
                if current_system_prompt.strip():
                    SYSTEM_PROMPT = current_system_prompt.strip()
                else:
                    SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
                
                # End current session and start new one
                pipe.finish_chat()
                pipe.start_chat(SYSTEM_PROMPT)
                print("🔄 Chat session reset with updated system prompt")
                return [], "", SYSTEM_PROMPT
            except Exception as e:
                print(f"⚠️ Session reset error: {e}")
                return [], "", current_system_prompt
        
        def show_metrics():
            """Display comprehensive performance metrics"""
            base_metrics = system_metrics.to_display_dict()
            
            qwen3_specific = {}
            if ENHANCED_CONTEXT_AVAILABLE:
                qwen3_specific = {
                    "Special Tokens Filtered": system_metrics.special_tokens_filtered,
                    "Compilation Errors": system_metrics.compilation_errors,
                    "NPUW Profile": NPU_OPTIMIZATION_PROFILE,
                    "Enhanced Features": "Active",
                    "Qwen3 Architecture": f"{QWEN3_8B_ARCHITECTURE.get('parameters', '8B')} parameters",
                    "Max Context": f"{QWEN3_8B_ARCHITECTURE.get('max_position_embeddings', 40960):,} tokens"
                }
            else:
                qwen3_specific = {
                    "Enhanced Features": "Fallback Mode",
                    "Note": "Install enhanced context for full optimization"
                }
            
            return (
                gr.update(value=base_metrics, visible=True),
                gr.update(value=qwen3_specific, visible=True) if ENHANCED_CONTEXT_AVAILABLE else gr.update(visible=False),
                gr.update(visible=True)
            )
        
        def show_system_info():
            """Display comprehensive system information"""
            info_text = f"""
            ## 🖥️ System Configuration
            
            **Hardware & Device:**
            - Target Device: {device_used}
            - Configuration: {config_used}
            - Cache Directory: `{CACHE_DIR}`
            - NPU Profile: {NPU_OPTIMIZATION_PROFILE}
            
            **Model Details:**
            - Model: Qwen3-8B INT4 Quantized
            - Path: `{MODEL_PATH}`
            - Load Time: {load_time:.1f} seconds
            - Tokenizer: {tokenizer.__class__.__name__}
            
            **OpenVINO GenAI Configuration:**
            - API Mode: Stateful (start_chat/finish_chat)
            - Conversation Management: Automatic KV-cache
            - Token Limits: {MAX_CONVERSATION_TOKENS} (conversation), {MAX_MESSAGE_LENGTH} (message)
            - Generation: Temperature={"0.6" if ENHANCED_CONTEXT_AVAILABLE else "0.7"}, Top-p={"0.95" if ENHANCED_CONTEXT_AVAILABLE else "0.9"}
            
            **Enhanced Features:**
            {"✅ Complete Qwen3 NPUW optimization" if ENHANCED_CONTEXT_AVAILABLE else "⚠️ Basic NPUW configuration"}
            {"✅ 26+ special token filtering" if ENHANCED_CONTEXT_AVAILABLE else "⚠️ Basic token filtering"}
            {"✅ Qwen3-specific chat templates" if ENHANCED_CONTEXT_AVAILABLE else "⚠️ Standard templates"}
            {"✅ Advanced performance monitoring" if ENHANCED_CONTEXT_AVAILABLE else "⚠️ Basic metrics"}
            
            **Performance Targets (NPU):**
            - Load Time: <90s (first run), <30s (cached)
            - First Token: <2s latency
            - Generation: 15-25 tokens/second
            - Memory: Optimized for NPU constraints
            """
            
            gr.Info(info_text)
        
        def reset_metrics():
            """Reset performance metrics"""
            global system_metrics
            session_start = system_metrics.session_start_time
            device = system_metrics.device_used
            config = system_metrics.config_used
            load_time = system_metrics.model_load_time
            
            system_metrics = SystemMetrics(
                session_start_time=session_start,
                device_used=device,
                config_used=config,
                model_load_time=load_time
            )
            
            gr.Info("📊 Performance metrics reset successfully")
        
        def reset_system_prompt():
            """Reset system prompt to default"""
            return DEFAULT_SYSTEM_PROMPT
        
        def apply_system_prompt(new_prompt):
            """Apply new system prompt and clear chat"""
            global SYSTEM_PROMPT
            if new_prompt.strip():
                SYSTEM_PROMPT = new_prompt.strip()
            else:
                SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
            
            try:
                pipe.finish_chat()
                pipe.start_chat(SYSTEM_PROMPT)
                gr.Info("✅ System prompt updated and chat cleared")
                return [], "", SYSTEM_PROMPT
            except Exception as e:
                gr.Warning(f"⚠️ Error applying prompt: {e}")
                return [], "", new_prompt
        
        def handle_file_upload(files):
            """Handle uploaded files for RAG processing"""
            if not files:
                return "No files selected"
            
            results = []
            for file in files:
                if file is None:
                    continue
                
                file_name = os.path.basename(file.name)
                result = rag_system.process_uploaded_file(file.name, file_name)
                results.append(result)
            
            return "\n\n".join(results)
        
        def clear_documents():
            """Clear all uploaded documents"""
            result = rag_system.clear_documents()
            gr.Info(result)
            return result
        
        def show_rag_status():
            """Show RAG system status"""
            status = rag_system.get_status()
            gr.Info(f"RAG Status: {status}")
            return str(status)
        
        # Wire up event handlers
        msg_input.submit(handle_send, [msg_input, chatbot], chatbot).then(
            lambda: gr.update(value=""), None, [msg_input]
        )
        
        send_btn.click(handle_send, [msg_input, chatbot], chatbot).then(
            lambda: gr.update(value=""), None, [msg_input]
        )
        
        clear_btn.click(handle_clear, [system_prompt_input], [chatbot, msg_input, system_prompt_input])
        
        metrics_btn.click(
            show_metrics, 
            None, 
            [metrics_json, qwen3_stats if ENHANCED_CONTEXT_AVAILABLE else None, metrics_row]
        )
        
        system_btn.click(show_system_info, None, None)
        reset_metrics_btn.click(reset_metrics, None, None)
        
        # System prompt event handlers
        reset_prompt_btn.click(reset_system_prompt, None, [system_prompt_input])
        apply_prompt_btn.click(
            apply_system_prompt, 
            [system_prompt_input], 
            [chatbot, msg_input, system_prompt_input]
        )
        
        # RAG event handlers
        file_upload.upload(handle_file_upload, [file_upload], [upload_status])
        clear_docs_btn.click(clear_documents, None, [upload_status])
        rag_status_btn.click(show_rag_status, None, [upload_status])
        
        # Initialize chat session when interface loads
        def initialize_session():
            """Initialize chat session with system prompt"""
            try:
                pipe.start_chat(SYSTEM_PROMPT)
                print("✅ Chat session initialized with system prompt")
            except Exception as e:
                print(f"⚠️ Session initialization error: {e}")
        
        demo.load(initialize_session, None, None)
    
    return demo

# --- Launch Application ---
if __name__ == "__main__":
    print("🌐 Launching Enhanced Qwen3 Chat Interface...")
    print("✨ Features:")
    print("   🎯 Complete Qwen3 NPUW optimization")
    print("   🔍 26+ special token filtering")
    print("   📊 Professional performance monitoring") 
    print("   🛡️ Robust error handling and diagnostics")
    print("   🔧 Official Gradio patterns integration")
    print("   💡 Intelligent message processing")
    print("=" * 60)
    
    demo = create_enhanced_interface()
    
    # Security-conscious launch configuration
    launch_config = {
        "share": False,  # Security: Never share publicly by default
        "server_name": "127.0.0.1",  # Security: Bind to localhost only
        "server_port": 7860,
        "show_error": True,
        "show_tips": True,
        "quiet": False,
        "auth": None,  # No authentication by default (add if needed)
        "max_file_size": "10mb",  # Limit file upload size
        "allowed_paths": []  # No file access by default
    }
    
    # Security warning if share is enabled via environment
    if os.getenv("GRADIO_SHARE", "").lower() in ('true', '1', 'yes'):
        print("⚠️ WARNING: Public sharing enabled via GRADIO_SHARE environment variable")
        print("🔒 Ensure your system is secure and consider adding authentication")
        launch_config["share"] = True
    
    demo.queue(
        max_size=20,
        default_concurrency_limit=1  # NPU works best with single concurrent requests
    ).launch(**launch_config)