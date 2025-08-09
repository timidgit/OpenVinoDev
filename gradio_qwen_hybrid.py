#!/usr/bin/env python3
"""
Hybrid Gradio Qwen Chat - Best of Both Worlds
=============================================

This hybrid version combines:
- Comprehensive fallback logic from gradio_qwen_debug.py (stability)
- RAG-inspired optimizations from gradio_qwen_optimized.py (performance)
- Consultant-recommended architectural improvements (maintainability)

Features:
- Exhaustive configuration cascade for maximum compatibility
- Semantic-aware truncation with punctuation breaks
- Rolling performance metrics and throughput monitoring
- Enhanced streaming with throttling options
- Professional error handling with detailed diagnostics
- Mode switching between development (debug) and production (optimized)
"""

import gradio as gr
import openvino_genai as ov_genai
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
from transformers import AutoTokenizer
import time
import queue
from threading import Thread
import re
from typing import Optional, Tuple, List, Dict, Any

# --- Configuration Constants ---
print("🚀 Starting Hybrid Qwen Chat System...")
model_path = r"C:\OpenVinoModels\qwen3-8b-int4-cw-ov"
device = "NPU"  # Can be switched to "CPU" for comparison

# Operational mode: "debug" for development, "production" for deployment
OPERATION_MODE = "debug"  # Change to "production" for cleaner output

# Named constants for maintainability
MAX_CONVERSATION_TOKENS = 1024
EMERGENCY_TOKEN_LIMIT = 1500
BASIC_TOKEN_LIMIT = 800
MAX_MESSAGE_LENGTH = 200
TRUNCATION_PREVIEW_LENGTH = 100
TRUNCATION_SUFFIX_LENGTH = 50
STREAMING_DELAY_MS = 30  # Throttle streaming for smoother UX
PERFORMANCE_WINDOW_SIZE = 15  # Rolling window for metrics

def log_debug(message: str):
    """Debug logging that respects operation mode"""
    if OPERATION_MODE == "debug":
        print(f"🔧 DEBUG: {message}")

def log_info(message: str):
    """Info logging for both modes"""
    print(f"ℹ️  {message}")

def log_error(message: str):
    """Error logging for both modes"""
    print(f"❌ ERROR: {message}")

def log_success(message: str):
    """Success logging for both modes"""
    print(f"✅ {message}")

# --- Advanced Configuration Factory ---
class ConfigurationManager:
    """Manages OpenVINO configurations with comprehensive fallback strategies"""
    
    @staticmethod
    def get_base_config() -> Dict[str, Any]:
        """Base configuration that works across all devices"""
        return {
            hints.performance_mode: hints.PerformanceMode.LATENCY,
            props.cache_dir: r"C:\temp\.ovcache"  # Cloud-friendly cache
        }
    
    @staticmethod
    def get_npu_config_cascade() -> List[Tuple[str, Dict[str, Any]]]:
        """Complete NPU configuration cascade from most to least optimized"""
        base = ConfigurationManager.get_base_config()
        
        # Configuration cascade - try from most optimized to most compatible
        configs = [
            # Tier 1: Full NPUW with advanced optimizations
            ("full_npuw_advanced", {
                **base,
                "NPU_USE_NPUW": "YES",
                "NPUW_LLM": "YES",
                "NPUW_LLM_BATCH_DIM": 0,
                "NPUW_LLM_SEQ_LEN_DIM": 1,
                "NPUW_LLM_MAX_PROMPT_LEN": 2048,
                "NPUW_LLM_MIN_RESPONSE_LEN": 256,
                "CACHE_MODE": "OPTIMIZE_SPEED",
                "INFERENCE_PRECISION_HINT": "f16",
                "MAX_PROMPT_LEN": 2048,
                "MIN_RESPONSE_LEN": 256
            }),
            
            # Tier 2: Basic NPUW without advanced properties
            ("basic_npuw", {
                **base,
                "NPU_USE_NPUW": "YES",
                "NPUW_LLM": "YES",
                "NPUW_LLM_MAX_PROMPT_LEN": 1024,
                "NPUW_LLM_MIN_RESPONSE_LEN": 128,
                "MAX_PROMPT_LEN": 1024,
                "MIN_RESPONSE_LEN": 128
            }),
            
            # Tier 3: NPU-specific optimizations without NPUW
            ("npu_specific", {
                **base,
                "CACHE_MODE": "OPTIMIZE_SPEED",
                "MAX_PROMPT_LEN": 1024
            }),
            
            # Tier 4: Base configuration with NPU device properties
            ("device_properties", base),
            
            # Tier 5: Minimal - no additional config
            ("minimal", {})
        ]
        
        return configs
    
    @staticmethod
    def get_cpu_config_cascade() -> List[Tuple[str, Dict[str, Any]]]:
        """CPU configuration cascade"""
        base = ConfigurationManager.get_base_config()
        
        return [
            ("cpu_optimized", {
                **base,
                props.streams.num: 1,
                props.inference_num_threads: 4,
                "MAX_PROMPT_LEN": 2048
            }),
            ("cpu_basic", base),
            ("cpu_minimal", {})
        ]

def try_load_pipeline_cascade(model_path: str, target_device: str) -> Tuple[Optional[ov_genai.LLMPipeline], str, str]:
    """
    Comprehensive pipeline loading with full fallback cascade
    Returns: (pipeline, device_used, config_used)
    """
    
    # Get appropriate configuration cascade
    if target_device == "NPU":
        primary_configs = ConfigurationManager.get_npu_config_cascade()
        fallback_device = "CPU"
        fallback_configs = ConfigurationManager.get_cpu_config_cascade()
    else:
        primary_configs = ConfigurationManager.get_cpu_config_cascade()
        fallback_device = None
        fallback_configs = []
    
    # Try primary device with all configurations
    log_info(f"Attempting to load model on {target_device}")
    for config_name, config in primary_configs:
        try:
            log_debug(f"Trying {target_device} with {config_name} configuration")
            
            if config:
                pipe = ov_genai.LLMPipeline(model_path, target_device, **config)
            else:
                pipe = ov_genai.LLMPipeline(model_path, target_device)
                
            log_success(f"Successfully loaded on {target_device} with {config_name} configuration")
            return pipe, target_device, config_name
            
        except Exception as e:
            log_debug(f"{target_device}/{config_name} failed: {e}")
            continue
    
    # Try fallback device if primary failed
    if fallback_device and fallback_configs:
        log_info(f"Primary device {target_device} failed, trying fallback {fallback_device}")
        
        for config_name, config in fallback_configs:
            try:
                log_debug(f"Trying {fallback_device} with {config_name} configuration")
                
                if config:
                    pipe = ov_genai.LLMPipeline(model_path, fallback_device, **config)
                else:
                    pipe = ov_genai.LLMPipeline(model_path, fallback_device)
                    
                log_success(f"Successfully loaded on {fallback_device} with {config_name} configuration")
                return pipe, fallback_device, config_name
                
            except Exception as e:
                log_debug(f"{fallback_device}/{config_name} failed: {e}")
                continue
    
    log_error("All device and configuration combinations failed")
    return None, "", ""

# --- Enhanced Generation Configuration ---
def create_generation_config(tokenizer: AutoTokenizer, mode: str = "balanced") -> ov_genai.GenerationConfig:
    """Create generation configuration based on operational mode"""
    config = ov_genai.GenerationConfig()
    
    if mode == "speed":
        # Speed-optimized for debug/testing
        config.do_sample = False
        config.max_new_tokens = 512
        config.repetition_penalty = 1.1
    elif mode == "quality":
        # Quality-optimized for production
        config.do_sample = True
        config.temperature = 0.7
        config.top_p = 0.9
        config.top_k = 50
        config.max_new_tokens = 1024
        config.repetition_penalty = 1.15
    else:  # balanced
        config.do_sample = True
        config.temperature = 0.6
        config.top_p = 0.85
        config.top_k = 40
        config.max_new_tokens = 768
        config.repetition_penalty = 1.12
    
    # Note: pad_token_id is handled by tokenizer in 2025 API, not GenerationConfig
    
    return config

# --- Semantic-Aware Text Processing ---
def semantic_aware_truncation(text: str, max_length: int) -> str:
    """Advanced truncation preserving semantic boundaries"""
    if len(text) <= max_length:
        return text
    
    # Try sentence boundaries first
    sentences = text.split('. ')
    if len(sentences) > 1:
        current_length = 0
        result_sentences = []
        
        for sentence in sentences:
            if current_length + len(sentence) + 2 <= max_length * 0.8:  # Leave room for ellipsis
                result_sentences.append(sentence)
                current_length += len(sentence) + 2
            else:
                break
        
        if result_sentences:
            remaining_text = '. '.join(sentences[len(result_sentences):])
            result = '. '.join(result_sentences) + '.'
            
            if len(remaining_text) > 30:
                preview = remaining_text[:30].strip()
                result += f" ... [continuing: {preview}...]"
            
            return result
    
    # Try phrase boundaries (punctuation)
    phrase_pattern = r'[,.;:!?]\s+'
    matches = list(re.finditer(phrase_pattern, text[:max_length]))
    
    if matches:
        best_break = max(match.end() for match in matches)
        if best_break > max_length * 0.6:
            prefix = text[:best_break].strip()
            suffix_start = best_break
            suffix_end = min(len(text), suffix_start + TRUNCATION_SUFFIX_LENGTH)
            suffix = text[suffix_start:suffix_end].strip()
            
            return f"{prefix} ... [truncated: {suffix}...]"
    
    # Fallback: clean word boundary truncation
    truncated = text[:TRUNCATION_PREVIEW_LENGTH]
    last_space = truncated.rfind(' ')
    
    if last_space > TRUNCATION_PREVIEW_LENGTH * 0.7:
        truncated = truncated[:last_space]
    
    suffix = text[-TRUNCATION_SUFFIX_LENGTH:].strip()
    return f"{truncated} ... [content truncated] ... {suffix}"

def intelligent_conversation_management(conversation: List[Dict], tokenizer: AutoTokenizer, 
                                      max_tokens: int, config_mode: str) -> List[Dict]:
    """Advanced conversation management with context preservation"""
    
    if len(conversation) <= 2:  # Keep system + current user
        return conversation
    
    iteration_count = 0
    max_iterations = 10  # Prevent infinite loops
    
    while len(conversation) > 2 and iteration_count < max_iterations:
        # Calculate token count
        try:
            test_prompt = tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            token_count = len(tokenizer.encode(test_prompt))
        except Exception as e:
            log_debug(f"Token counting failed: {e}, using conservative estimate")
            # Conservative estimate: ~4 chars per token
            char_count = sum(len(msg.get('content', '')) for msg in conversation)
            token_count = char_count // 4
        
        log_debug(f"Conversation management: {token_count} tokens, iteration {iteration_count}")
        
        if token_count <= max_tokens:
            break
        
        # Smart removal strategy
        if len(conversation) >= 4:  # Remove user-assistant pairs
            removed_user = conversation.pop(1)
            removed_assistant = conversation.pop(1) if len(conversation) > 1 else None
            
            user_chars = len(removed_user.get('content', ''))
            assistant_chars = len(removed_assistant.get('content', '')) if removed_assistant else 0
            
            if OPERATION_MODE == "debug":
                log_debug(f"Removed exchange: {user_chars} + {assistant_chars} chars")
                
        elif len(conversation) > 2:  # Remove single message
            removed = conversation.pop(1)
            if OPERATION_MODE == "debug":
                log_debug(f"Removed message: {len(removed.get('content', ''))} chars")
        else:
            break
            
        iteration_count += 1
    
    if iteration_count >= max_iterations:
        log_debug("Conversation management hit max iterations, proceeding anyway")
    
    return conversation

# --- Enhanced Streaming with Performance Optimization ---
class HybridStreamer(ov_genai.StreamerBase):
    """Advanced streaming with performance monitoring and smooth UX"""
    
    def __init__(self, tokenizer: AutoTokenizer, enable_throttling: bool = True):
        super().__init__()
        self.tokenizer = tokenizer
        self.q = queue.Queue()
        self.accumulated_tokens = []
        self.full_response = ""
        self.enable_throttling = enable_throttling
        self.token_count = 0
        self.start_time = time.time()
    
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning with space preservation"""
        special_tokens = [
            '<|im_start|>', '<|im_end|>', 
            '<|system|>', '<|user|>', '<|assistant|>',
            '<|endoftext|>', '<|end|>', '<|start|>',
            '</s>', '<s>', '[INST]', '[/INST]',
            '<pad>', '<unk>', '<mask>'
        ]
        
        cleaned = text
        for token in special_tokens:
            cleaned = cleaned.replace(token, '')
        
        # Normalize whitespace but preserve structure
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def put(self, token_id: int) -> bool:
        self.accumulated_tokens.append(token_id)
        self.token_count += 1
        
        try:
            decoded_text = self.tokenizer.decode(self.accumulated_tokens, skip_special_tokens=True)
        except Exception as e:
            log_debug(f"Decoding error: {e}")
            decoded_text = self.tokenizer.decode(self.accumulated_tokens)
        
        if len(decoded_text) > len(self.full_response):
            new_text = decoded_text[len(self.full_response):]
            self.full_response = decoded_text
            
            if new_text.strip():
                cleaned_text = self.clean_text(new_text)
                if cleaned_text:
                    self.q.put(cleaned_text)
                    
                    # Throttling for smoother UX
                    if self.enable_throttling and STREAMING_DELAY_MS > 0:
                        time.sleep(STREAMING_DELAY_MS / 1000.0)
        
        return False
    
    def end(self):
        elapsed = time.time() - self.start_time
        tokens_per_sec = self.token_count / elapsed if elapsed > 0 else 0
        log_debug(f"Streaming completed: {self.token_count} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
        self.q.put(None)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        item = self.q.get()
        if item is None:
            raise StopIteration
        return item

# --- Performance Monitoring System ---
class AdvancedPerformanceTracker:
    """Comprehensive performance tracking with multiple metrics"""
    
    def __init__(self, window_size: int = PERFORMANCE_WINDOW_SIZE):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        self.total_requests = 0
        self.response_times = []
        self.token_counts = []
        self.input_token_counts = []
        self.error_counts = {"token_limit": 0, "model": 0, "other": 0}
        self.configuration = "unknown"
        self.device_used = "unknown"
        self.start_time = time.time()
    
    def update_request(self, response_time: float, input_tokens: int, output_tokens: int, 
                      config_type: str, device: str, error_type: str = None):
        self.total_requests += 1
        self.response_times.append(response_time)
        self.token_counts.append(output_tokens)
        self.input_token_counts.append(input_tokens)
        self.configuration = config_type
        self.device_used = device
        
        if error_type:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Maintain rolling window
        if len(self.response_times) > self.window_size:
            self.response_times.pop(0)
            self.token_counts.pop(0)
            self.input_token_counts.pop(0)
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        if not self.response_times:
            return self._empty_metrics()
        
        # Calculate averages
        avg_response_time = sum(self.response_times) / len(self.response_times)
        avg_output_tokens = sum(self.token_counts) / len(self.token_counts)
        avg_input_tokens = sum(self.input_token_counts) / len(self.input_token_counts)
        
        # Calculate throughput
        total_tokens = sum(self.token_counts)
        total_time = sum(self.response_times)
        throughput = total_tokens / total_time if total_time > 0 else 0
        
        # Session statistics
        session_time = time.time() - self.start_time
        total_errors = sum(self.error_counts.values())
        success_rate = ((self.total_requests - total_errors) / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "session": {
                "total_requests": self.total_requests,
                "session_duration": round(session_time, 1),
                "success_rate": round(success_rate, 1),
                "total_errors": total_errors
            },
            "performance": {
                "avg_response_time": round(avg_response_time, 2),
                "avg_input_tokens": round(avg_input_tokens, 1),
                "avg_output_tokens": round(avg_output_tokens, 1),
                "throughput_tokens_per_sec": round(throughput, 1),
                "last_response_time": round(self.response_times[-1], 2) if self.response_times else 0
            },
            "system": {
                "device": self.device_used,
                "configuration": self.configuration,
                "rolling_window_size": len(self.response_times),
                "operation_mode": OPERATION_MODE
            },
            "errors": dict(self.error_counts)
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "session": {"total_requests": 0, "session_duration": 0, "success_rate": 100, "total_errors": 0},
            "performance": {"avg_response_time": 0, "avg_input_tokens": 0, "avg_output_tokens": 0, "throughput_tokens_per_sec": 0, "last_response_time": 0},
            "system": {"device": "unknown", "configuration": "unknown", "rolling_window_size": 0, "operation_mode": OPERATION_MODE},
            "errors": {"token_limit": 0, "model": 0, "other": 0}
        }

# --- Initialize System ---
log_info(f"Model path: {model_path}")
log_info(f"Target device: {device}")
log_info(f"Operation mode: {OPERATION_MODE}")

print("\n⏳ Loading model with comprehensive fallback system...")
load_start_time = time.time()

# Load pipeline with cascade
pipe, device_used, config_used = try_load_pipeline_cascade(model_path, device)

if not pipe:
    log_error("Failed to load model with any configuration. Please check:")
    log_error("1. Model path exists and is valid")
    log_error("2. OpenVINO installation is correct")
    log_error("3. Device drivers are properly installed")
    exit(1)

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Configure tokenizer
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    log_success("Tokenizer loaded and configured")
    
except Exception as e:
    log_error(f"Failed to load tokenizer: {e}")
    exit(1)

# Create generation configuration
generation_mode = "quality" if OPERATION_MODE == "production" else "balanced"
generation_config = create_generation_config(tokenizer, generation_mode)

# Initialize performance tracker
performance_tracker = AdvancedPerformanceTracker()

load_end_time = time.time()
log_success(f"System initialized in {load_end_time - load_start_time:.2f} seconds")
log_info(f"Using device: {device_used} with configuration: {config_used}")

# --- System Prompt ---
system_prompt = (
    "You are a helpful AI assistant powered by OpenVINO on NPU. "
    "Provide accurate, concise responses while being friendly and informative. "
    "If asked about technical topics, explain them clearly and provide practical examples when helpful."
)

# --- Main Chat Function ---
def hybrid_chat_function(message: str, history: list):
    """
    Hybrid chat function combining robust error handling with performance optimization
    """
    request_start_time = time.time()
    input_token_count = 0
    output_token_count = 0
    error_type = None
    
    try:
        # Input validation
        if not message or not message.strip():
            yield history
            return
            
        if len(message) > 2000:  # Reasonable limit
            message = semantic_aware_truncation(message, 1500)
            log_debug("Input message truncated due to length")
        
        # Build conversation
        conversation = [{"role": "system", "content": system_prompt}] + history
        conversation.append({"role": "user", "content": message})
        
        # Determine token limits based on configuration
        if "full_npuw" in config_used or "advanced" in config_used:
            max_conversation_tokens = MAX_CONVERSATION_TOKENS * 2  # Higher limit for advanced configs
            emergency_limit = EMERGENCY_TOKEN_LIMIT * 2
        else:
            max_conversation_tokens = MAX_CONVERSATION_TOKENS
            emergency_limit = EMERGENCY_TOKEN_LIMIT
        
        # Intelligent conversation management
        conversation = intelligent_conversation_management(
            conversation, tokenizer, max_conversation_tokens, config_used
        )
        
        # Generate prompt
        try:
            prompt = tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            input_token_count = len(tokenizer.encode(prompt))
            
        except Exception as e:
            log_error(f"Failed to apply chat template: {e}")
            error_type = "other"
            yield history + [{"role": "assistant", "content": "❌ Error processing conversation format. Please try starting a new chat."}]
            return
        
        # Emergency truncation check
        if input_token_count > emergency_limit:
            log_debug(f"Emergency truncation needed: {input_token_count} > {emergency_limit}")
            
            # Try semantic truncation of user message
            if len(message) > MAX_MESSAGE_LENGTH:
                truncated_message = semantic_aware_truncation(message, MAX_MESSAGE_LENGTH)
                conversation[-1]["content"] = truncated_message
                
                try:
                    prompt = tokenizer.apply_chat_template(
                        conversation, add_generation_prompt=True, tokenize=False
                    )
                    input_token_count = len(tokenizer.encode(prompt))
                    log_debug(f"After emergency truncation: {input_token_count} tokens")
                    
                except Exception as e:
                    log_error(f"Emergency truncation failed: {e}")
                    error_type = "token_limit"
                    yield history + [{"role": "assistant", "content": "❌ Message too long for current configuration. Please try a shorter message."}]
                    return
        
        # Update history for UI
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        
        # Initialize streaming
        streamer = HybridStreamer(tokenizer, enable_throttling=(OPERATION_MODE == "production"))
        
        # Generate response in separate thread
        def generate():
            try:
                pipe.generate(prompt, generation_config, streamer)
            except Exception as e:
                log_error(f"Generation error: {e}")
                streamer.q.put(f"❌ Generation failed: {str(e)[:100]}...")
                streamer.q.put(None)
        
        generation_thread = Thread(target=generate, daemon=True)
        generation_thread.start()
        
        # Stream response to UI
        try:
            for new_text in streamer:
                if new_text:
                    history[-1]["content"] += new_text
                    output_token_count = len(tokenizer.encode(history[-1]["content"]))
                    yield history
                    
        except Exception as e:
            log_error(f"Streaming error: {e}")
            error_type = "other"
            
        generation_thread.join(timeout=30.0)  # Prevent hanging
        
        if generation_thread.is_alive():
            log_error("Generation thread timed out")
            error_type = "other"
        
    except Exception as e:
        log_error(f"Chat function error: {e}")
        error_type = "other"
        
        # Categorize error for better user feedback
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["token", "length", "sequence"]):
            error_type = "token_limit"
            user_msg = "⚠️ Input too long for current model configuration. Try a shorter message or start a new chat."
        elif any(keyword in error_msg for keyword in ["memory", "resource", "allocation"]):
            user_msg = "⚠️ System resources exceeded. Please start a new conversation."
        else:
            user_msg = f"❌ Unexpected error occurred. Error details: {str(e)[:150]}..."
        
        if not history or history[-1]["role"] != "assistant":
            history.append({"role": "assistant", "content": user_msg})
        else:
            history[-1]["content"] = user_msg
            
        yield history
    
    finally:
        # Update performance metrics
        request_time = time.time() - request_start_time
        performance_tracker.update_request(
            request_time, input_token_count, output_token_count, 
            config_used, device_used, error_type
        )
        
        # Log performance for debug mode
        if OPERATION_MODE == "debug":
            metrics = performance_tracker.get_comprehensive_metrics()
            perf = metrics["performance"]
            log_debug(f"Request: {request_time:.2f}s, {input_token_count}→{output_token_count} tokens, {perf['throughput_tokens_per_sec']:.1f} tok/s")

# --- Gradio Interface ---
def create_gradio_interface():
    """Create the hybrid Gradio interface"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Hybrid Qwen Chat",
        css="""
        .gradio-container { max-width: 1400px; margin: auto; }
        .chatbot { height: 650px; }
        .performance-panel { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }
        """,
        fill_height=True
    ) as demo:
        
        # Header
        gr.Markdown(f"""
        # 🤖 Hybrid Qwen Chat System
        
        **Status**: ✅ Loaded on **{device_used}** using **{config_used}** configuration  
        **Model**: Qwen3-8B (INT4 Channel-wise)  
        **Mode**: {OPERATION_MODE.title()} mode with comprehensive fallback system  
        **Generation**: {generation_mode.title()} quality settings
        """)
        
        # Main chat interface
        chatbot = gr.Chatbot(
            label="Conversation", 
            height=650, 
            type='messages',
            avatar_images=(None, "🤖"),
            show_copy_button=True,
            show_share_button=False
        )
        
        # Input section
        with gr.Row():
            msg = gr.Textbox(
                label="Your Message", 
                placeholder=f"Ask me anything... (Mode: {OPERATION_MODE}, Device: {device_used})", 
                scale=8,
                max_lines=4,
                show_label=False
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Control buttons
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            info_btn = gr.Button("ℹ️ System Info", variant="secondary") 
            perf_btn = gr.Button("📊 Performance", variant="secondary")
            mode_btn = gr.Button("🔧 Toggle Debug", variant="secondary")
        
        # Performance panel (collapsible)
        with gr.Row(visible=False) as perf_row:
            with gr.Column():
                perf_display = gr.JSON(label="Performance Metrics", visible=False)
        
        # System info panel (collapsible)  
        with gr.Row(visible=False) as info_row:
            with gr.Column():
                info_display = gr.Markdown(visible=False)
        
        # Event handlers
        msg.submit(hybrid_chat_function, [msg, chatbot], chatbot, queue=True).then(
            lambda: gr.update(value=""), None, [msg], queue=False
        )
        
        submit_btn.click(hybrid_chat_function, [msg, chatbot], chatbot, queue=True).then(
            lambda: gr.update(value=""), None, [msg], queue=False
        )
        
        clear_btn.click(
            lambda: ([], ""), None, [chatbot, msg], queue=False
        ).then(
            lambda: performance_tracker.reset(), None, None, queue=False
        )
        
        # System info function
        def show_system_info():
            metrics = performance_tracker.get_comprehensive_metrics()
            
            info = f"""
            ## 🖥️ System Configuration
            - **Device**: {device_used}
            - **Configuration**: {config_used}
            - **Model Path**: `{model_path}`
            - **Operation Mode**: {OPERATION_MODE}
            - **Generation Mode**: {generation_mode}
            
            ## ⚙️ Current Settings  
            - **Max Conversation Tokens**: {MAX_CONVERSATION_TOKENS}
            - **Emergency Token Limit**: {EMERGENCY_TOKEN_LIMIT}
            - **Streaming Delay**: {STREAMING_DELAY_MS}ms
            - **Performance Window**: {PERFORMANCE_WINDOW_SIZE} requests
            
            ## 📊 Generation Config
            - **Sampling**: {"Enabled" if generation_config.do_sample else "Disabled"}
            - **Temperature**: {getattr(generation_config, 'temperature', 'N/A')}
            - **Top-p**: {getattr(generation_config, 'top_p', 'N/A')}
            - **Max Tokens**: {generation_config.max_new_tokens}
            - **Repetition Penalty**: {generation_config.repetition_penalty}
            
            ## 🔄 Session Statistics
            - **Total Requests**: {metrics['session']['total_requests']}
            - **Session Duration**: {metrics['session']['session_duration']}s  
            - **Success Rate**: {metrics['session']['success_rate']}%
            """
            
            return gr.update(value=info, visible=True), gr.update(visible=True)
        
        info_btn.click(show_system_info, None, [info_display, info_row])
        
        # Performance metrics function
        def show_performance():
            metrics = performance_tracker.get_comprehensive_metrics()
            return gr.update(value=metrics, visible=True), gr.update(visible=True)
        
        perf_btn.click(show_performance, None, [perf_display, perf_row])
        
        # Mode toggle function
        def toggle_debug_mode():
            global OPERATION_MODE
            OPERATION_MODE = "production" if OPERATION_MODE == "debug" else "debug"
            log_info(f"Switched to {OPERATION_MODE} mode")
            return gr.update(value=f"Mode: {OPERATION_MODE}")
        
        mode_btn.click(
            toggle_debug_mode, None, None
        ).then(
            lambda: gr.update(placeholder=f"Ask me anything... (Mode: {OPERATION_MODE}, Device: {device_used})"),
            None, msg
        )
        
        # Example prompts
        gr.Examples(
            examples=[
                "Explain how neural networks learn and adapt",
                "Write a Python function to implement binary search", 
                "What are the latest developments in AI hardware acceleration?",
                "Compare different optimization techniques for deep learning models",
                "How does OpenVINO optimize inference for different hardware?"
            ], 
            inputs=msg,
            label="Example Prompts"
        )
        
        return demo

# --- Launch Application ---
if __name__ == "__main__":
    log_info("🌐 Starting Hybrid Gradio Interface...")
    log_info("🎯 Hybrid system features:")
    log_info("   • Comprehensive device and configuration fallback cascade")
    log_info("   • Semantic-aware truncation with punctuation boundary detection")
    log_info("   • Advanced performance monitoring with rolling metrics") 
    log_info("   • Enhanced streaming with configurable throttling")
    log_info("   • Professional error handling with categorized user feedback")
    log_info("   • Switchable debug/production modes")
    
    demo = create_gradio_interface()
    
    try:
        demo.queue(
            max_size=20,
            default_concurrency_limit=3
        ).launch(
            share=False,
            server_name="127.0.0.1", 
            server_port=7860,
            show_error=True,
            quiet=(OPERATION_MODE == "production")
        )
    except KeyboardInterrupt:
        log_info("🛑 Application stopped by user")
    except Exception as e:
        log_error(f"Application error: {e}")