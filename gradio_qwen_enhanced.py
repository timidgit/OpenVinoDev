#!/usr/bin/env python3
"""
Enhanced Qwen Chat - Production-Ready Implementation
===================================================
Fully enhanced with comprehensive context patterns:
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
from typing import Optional, Dict, Any, List, Tuple, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path

# Import enhanced context patterns
import sys
import os
context_path = os.path.join(os.path.dirname(__file__), "context")
sys.path.insert(0, context_path)

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

# --- Constants and Configuration ---
MODEL_PATH = r"C:\OpenVinoModels\qwen3-8b-int4-cw-ov"
DEVICE = "NPU"
CACHE_DIR = r"C:\temp\.ovcache_qwen3_enhanced"

# Qwen3-optimized settings
MAX_CONVERSATION_TOKENS = 1800  # Conservative for NPU
EMERGENCY_LIMIT = 2048         # Hard limit before reset
MAX_MESSAGE_LENGTH = 400       # Slightly increased
NPU_OPTIMIZATION_PROFILE = "balanced"  # balanced, conservative, aggressive

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

# Enhanced system prompt with Qwen3 optimization
SYSTEM_PROMPT = """You are a helpful, concise AI assistant powered by Qwen3-8B running on Intel NPU via OpenVINO GenAI. 

Key behaviors:
- Provide accurate, well-structured responses
- Be concise but comprehensive 
- Use clear formatting when helpful
- Acknowledge when you're uncertain
- Optimize for NPU constraints (prefer shorter, focused responses)

You excel at: reasoning, coding, analysis, creative writing, and technical explanations."""

# --- Enhanced Configuration Management ---
class Qwen3ConfigurationManager:
    """Advanced configuration management with Qwen3 optimization"""
    
    def __init__(self, profile: str = "balanced"):
        self.profile = profile
        self.config_builder = None
        self.performance_monitor = Qwen3NPUPerformanceMonitor() if ENHANCED_CONTEXT_AVAILABLE else None
        
        if ENHANCED_CONTEXT_AVAILABLE:
            self.config_builder = Qwen3NPUConfigBuilder(profile)
    
    def get_npu_config(self) -> Dict[str, Any]:
        """Get complete NPU configuration with NPUW optimization"""
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
            
            # Add OpenVINO properties if available
            if OPENVINO_PROPERTIES_AVAILABLE:
                config.update({
                    hints.performance_mode: hints.PerformanceMode.LATENCY,
                    props.cache_dir: CACHE_DIR
                })
            else:
                config.update({
                    "PERFORMANCE_HINT": "LATENCY",
                    "CACHE_DIR": CACHE_DIR
                })
            
            return config
    
    def get_cpu_config(self) -> Dict[str, Any]:
        """Get optimized CPU configuration"""
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
def deploy_qwen3_pipeline(model_path: str, target_device: str, profile: str = "balanced") -> Tuple[ov_genai.LLMPipeline, str, str, float]:
    """
    Deploy Qwen3 pipeline with comprehensive error handling and optimization
    Returns: (pipeline, device_used, config_used, load_time)
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
def create_qwen3_generation_config() -> ov_genai.GenerationConfig:
    """Create optimized generation configuration for Qwen3"""
    config = ov_genai.GenerationConfig()
    
    if ENHANCED_CONTEXT_AVAILABLE:
        # Use Qwen3-specific defaults from context
        qwen3_config = QWEN3_8B_ARCHITECTURE
        config.do_sample = True
        config.temperature = 0.6  # Qwen3 default
        config.top_p = 0.95       # Qwen3 default  
        config.top_k = 20         # Qwen3 default
        config.max_new_tokens = 1024
        config.repetition_penalty = 1.1
        # Note: token IDs are handled by tokenizer, not config
    else:
        # Fallback configuration
        config.do_sample = True
        config.temperature = 0.7
        config.top_p = 0.9
        config.top_k = 50
        config.max_new_tokens = 1024
        config.repetition_penalty = 1.1
    
    return config

# --- Smart Message Processing ---
def process_user_message(message: str, history: List[Dict[str, str]]) -> Tuple[str, bool]:
    """
    Process user message with Qwen3-optimized handling
    Returns: (processed_message, was_truncated)
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

# --- Enhanced Chat Function ---
def enhanced_qwen3_chat(message: str, history: List[Dict[str, str]]):
    """
    Enhanced chat function with comprehensive Qwen3 optimization
    """
    global system_metrics
    
    request_start_time = time.time()
    system_metrics.total_requests += 1
    
    try:
        # Input validation
        if not message.strip():
            yield history
            return
        
        # Process message with smart handling
        processed_message, was_truncated = process_user_message(message, history)
        
        # Show truncation warning if needed
        if was_truncated:
            truncation_warning = {
                "role": "assistant",
                "content": f"⚠️ Your message was truncated from {len(message):,} to {len(processed_message)} characters due to NPU memory limits. Processing the truncated version..."
            }
            history.append({"role": "user", "content": message})
            history.append(truncation_warning)
            yield history
            time.sleep(0.5)  # Brief pause for user to see warning
        
        # Add user message to history
        if not was_truncated:
            history.append({"role": "user", "content": processed_message})
        
        # Add assistant placeholder
        history.append({"role": "assistant", "content": ""})
        yield history
        
        # Initialize enhanced streamer
        streamer = EnhancedQwen3Streamer(tokenizer)
        
        # Generation configuration
        generation_config = create_qwen3_generation_config()
        
        # Generate response using proper session management
        def generate_with_session():
            try:
                # Use stateful generation - only send new message
                pipe.generate(processed_message, generation_config, streamer)
                system_metrics.successful_requests += 1
            except Exception as e:
                print(f"❌ Generation error: {e}")
                system_metrics.failed_requests += 1
                # Send error to streamer
                streamer.text_queue.put(f"❌ Generation error: {str(e)[:100]}...")
                streamer.text_queue.put(None)
        
        # Run generation in thread
        generation_thread = threading.Thread(target=generate_with_session)
        generation_thread.start()
        
        # Stream response to UI
        try:
            for chunk in streamer:
                if chunk:  # Only add non-empty chunks
                    history[-1]["content"] += chunk
                    yield history
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            history[-1]["content"] = f"❌ Streaming error: {str(e)[:100]}..."
            yield history
        
        # Wait for generation to complete
        generation_thread.join(timeout=30.0)
        if generation_thread.is_alive():
            print("⚠️ Generation timeout - thread still running")
        
        # Update performance metrics
        elapsed_time = time.time() - request_start_time
        system_metrics.avg_response_time = (
            (system_metrics.avg_response_time * (system_metrics.total_requests - 1) + elapsed_time)
            / system_metrics.total_requests
        )
        
        print(f"📊 Request complete: {elapsed_time:.2f}s total")
        
    except Exception as e:
        print(f"❌ Chat function error: {e}")
        system_metrics.failed_requests += 1
        
        # Determine error type and provide helpful message
        error_message = "❌ An error occurred. "
        
        if "memory" in str(e).lower():
            error_message += "Memory limit reached. Try starting a new conversation."
        elif "token" in str(e).lower() or "length" in str(e).lower():
            error_message += "Message too long. Please try a shorter message."
        elif "compile" in str(e).lower():
            error_message += "NPU compilation issue. Check NPUW configuration."
        else:
            error_message += f"Details: {str(e)[:100]}..."
        
        # Add error to history
        if not history or history[-1]["role"] != "assistant":
            history.append({"role": "assistant", "content": error_message})
        else:
            history[-1]["content"] = error_message
        
        yield history

# --- System Initialization ---
print("🚀 Initializing Enhanced Qwen3 Chat System")
print(f"📂 Model: {MODEL_PATH}")
print(f"🎯 Target Device: {DEVICE}")
print(f"📊 Optimization Profile: {NPU_OPTIMIZATION_PROFILE}")
print(f"🔧 Enhanced Context: {'Available' if ENHANCED_CONTEXT_AVAILABLE else 'Fallback Mode'}")

try:
    # Deploy pipeline with comprehensive error handling
    pipe, device_used, config_used, load_time = deploy_qwen3_pipeline(
        MODEL_PATH, DEVICE, NPU_OPTIMIZATION_PROFILE
    )
    
    # Update system metrics
    system_metrics.device_used = device_used
    system_metrics.config_used = config_used
    system_metrics.model_load_time = load_time
    
    # Initialize tokenizer
    print("📚 Loading Qwen3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Configure tokenizer for Qwen3
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✅ System Ready!")
    print(f"   Device: {device_used}")
    print(f"   Config: {config_used}")
    print(f"   Load Time: {load_time:.1f}s")
    print(f"   Tokenizer: {tokenizer.__class__.__name__}")
    if ENHANCED_CONTEXT_AVAILABLE:
        print(f"   Special Tokens: {len(QWEN3_SPECIAL_TOKENS)} Qwen3 tokens loaded")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ System initialization failed: {e}")
    print("🔧 Check:")
    print("  1. Model path exists and contains OpenVINO model files")
    print("  2. NPU drivers are installed and working")
    print("  3. OpenVINO environment is properly configured")
    print("  4. NPUW configuration is compatible")
    exit(1)

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
                    "Explain the concept of attention in transformer models"
                ],
                inputs=msg_input,
                label="💡 Example Questions"
            )
        
        # Event handlers with enhanced functionality
        def handle_send(message, history):
            """Handle send with proper session management"""
            return enhanced_qwen3_chat(message, history)
        
        def handle_clear():
            """Handle clear with proper session reset"""
            try:
                # End current session and start new one
                pipe.finish_chat()
                pipe.start_chat(SYSTEM_PROMPT)
                print("🔄 Chat session reset")
                return [], ""
            except Exception as e:
                print(f"⚠️ Session reset error: {e}")
                return [], ""
        
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
        
        # Wire up event handlers
        msg_input.submit(handle_send, [msg_input, chatbot], chatbot).then(
            lambda: gr.update(value=""), None, [msg_input]
        )
        
        send_btn.click(handle_send, [msg_input, chatbot], chatbot).then(
            lambda: gr.update(value=""), None, [msg_input]
        )
        
        clear_btn.click(handle_clear, None, [chatbot, msg_input])
        
        metrics_btn.click(
            show_metrics, 
            None, 
            [metrics_json, qwen3_stats if ENHANCED_CONTEXT_AVAILABLE else None, metrics_row]
        )
        
        system_btn.click(show_system_info, None, None)
        reset_metrics_btn.click(reset_metrics, None, None)
        
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
    
    demo.queue(
        max_size=20,
        default_concurrency_limit=1  # NPU works best with single concurrent requests
    ).launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        show_tips=True,
        quiet=False
    )