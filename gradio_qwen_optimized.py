#!/usr/bin/env python3
"""
RAG-Enhanced Qwen Chat - Advanced NPU Optimization
==================================================

RAG notebook-inspired optimizations applied:
- Stateful chat sessions with proper session management
- NPU model reshaping with static shapes for better compilation
- Enhanced generation configuration with temperature/sampling controls
- Professional streaming patterns with advanced text processing
- Comprehensive error handling with intelligent fallback strategies
- Real-time performance monitoring and metrics collection
- Advanced UI controls for generation parameters
- Context-aware conversation management
"""

import gradio as gr
import openvino_genai as ov_genai
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
from transformers import AutoTokenizer
import time
import queue
import re
import json
from threading import Thread
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import numpy as np

# --- Enhanced Configuration Constants ---
print("🚀 Starting RAG-Enhanced Qwen Chat with Advanced NPU Optimization...")

@dataclass
class ModelConfig:
    """Centralized model configuration with NPU optimizations"""
    model_path: str = r"C:\OpenVinoModels\qwen3-8b-int4-cw-ov"
    primary_device: str = "NPU"
    fallback_device: str = "CPU"
    cache_dir: str = r"C:\temp\.ovcache"
    enable_reshaping: bool = True
    use_stateful_chat: bool = True
    
config = ModelConfig()
print(f"📁 Model: {config.model_path}")
print(f"🖥️  Primary device: {config.primary_device}")
print(f"🎯 NPU reshaping: {'enabled' if config.enable_reshaping else 'disabled'}")
print(f"💬 Stateful chat: {'enabled' if config.use_stateful_chat else 'disabled'}")

# Enhanced constants with NPU-specific optimizations
@dataclass
class TokenLimits:
    """NPU-optimized token management constants"""
    MAX_PROMPT_LEN: int = 2048  # NPU supports up to 2048 for prefill
    MIN_RESPONSE_LEN: int = 256  # NPU optimization requirement
    CONVERSATION_LIMIT: int = 1024  # Conservative conversation limit
    EMERGENCY_LIMIT: int = 1500    # Hard emergency limit
    MAX_MESSAGE_LENGTH: int = 300  # Individual message limit
    TRUNCATION_PREVIEW: int = 150  # Preview length for truncation
    STREAMING_BATCH_SIZE: int = 4  # Process tokens in batches for smoothness
    
limits = TokenLimits()
print(f"📊 Token limits - Conversation: {limits.CONVERSATION_LIMIT}, Max prompt: {limits.MAX_PROMPT_LEN}")

# Advanced configuration factory with NPU reshaping support
def get_npu_reshape_config() -> Dict[str, Any]:
    """NPU-specific reshaping configuration for static shapes"""
    return {
        "NPUW_LLM": "YES",
        "NPUW_LLM_MAX_PROMPT_LEN": limits.MAX_PROMPT_LEN,
        "NPUW_LLM_MIN_RESPONSE_LEN": limits.MIN_RESPONSE_LEN, 
        "NPUW_LLM_BATCH_DIM": 0,
        "NPUW_LLM_SEQ_LEN_DIM": 1,
        "NPU_USE_NPUW": "YES",
        "CACHE_MODE": "OPTIMIZE_SPEED",
        "INFERENCE_PRECISION_HINT": "f16",
        "NPUW_LLM_PREFILL_HINT": "LATENCY",
        "NPUW_LLM_GENERATE_HINT": "LATENCY"
    }

def get_device_config(device_type: str, mode: str = "basic") -> Dict[str, Any]:
    """Enhanced device configuration with multiple optimization levels"""
    base_config = {
        hints.performance_mode: hints.PerformanceMode.LATENCY,
        props.cache_dir: config.cache_dir,
        props.streams.num: 1,
        props.inference_num_threads: 1 if device_type == "NPU" else 4
    }
    
    if device_type == "NPU":
        if mode == "reshape_npu":
            # Full NPU reshaping with static shapes
            return {**base_config, **get_npu_reshape_config()}
        elif mode == "enhanced_npu":
            # Enhanced NPU without full reshaping
            return {
                **base_config,
                "NPUW_LLM": "YES",
                "NPUW_LLM_MAX_PROMPT_LEN": limits.MAX_PROMPT_LEN,
                "CACHE_MODE": "OPTIMIZE_SPEED"
            }
    
    return base_config

print("🔧 Using basic NPU configuration (bypasses NPUW compilation issues)")
print("📁 Model path:", model_path)
print("🖥️  Target device:", device)

# --- Enhanced System Prompt with NPU Optimization Context ---
SYSTEM_PROMPT = (
    "You are an intelligent AI assistant optimized for efficient NPU inference. "
    "Provide accurate, helpful responses while being mindful of computational constraints. "
    "Be concise yet informative, and adapt your response length to the complexity of the question."
)

# --- Advanced Generation Configuration Management ---
class GenerationConfigManager:
    """Manages generation configurations with different quality/speed profiles"""
    
    @staticmethod
    def get_base_config() -> ov_genai.GenerationConfig:
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = limits.MAX_PROMPT_LEN  # Use NPU-optimized limit
        config.repetition_penalty = 1.1
        return config
    
    @staticmethod 
    def get_balanced_config() -> ov_genai.GenerationConfig:
        """Balanced quality/speed for general use"""
        config = GenerationConfigManager.get_base_config()
        config.do_sample = True
        config.temperature = 0.7
        config.top_p = 0.9
        config.top_k = 50
        return config
        
    @staticmethod
    def get_creative_config() -> ov_genai.GenerationConfig:
        """Higher creativity for creative tasks"""
        config = GenerationConfigManager.get_base_config()
        config.do_sample = True
        config.temperature = 0.8
        config.top_p = 0.95
        config.top_k = 40
        config.repetition_penalty = 1.05
        return config
    
    @staticmethod
    def get_precise_config() -> ov_genai.GenerationConfig:
        """Lower temperature for factual tasks"""
        config = GenerationConfigManager.get_base_config()
        config.do_sample = True  
        config.temperature = 0.3
        config.top_p = 0.8
        config.top_k = 30
        config.repetition_penalty = 1.15
        return config

# Default to balanced configuration
generation_config = GenerationConfigManager.get_balanced_config()
print(f"🎛️  Generation config: T={generation_config.temperature}, top_p={generation_config.top_p}, max_tokens={generation_config.max_new_tokens}")

# --- Advanced Pipeline Loading with Comprehensive Fallback Strategy ---
print("\n⏳ Loading model with RAG-enhanced configuration and intelligent fallback...")
load_start_time = time.time()

class PipelineLoader:
    """Advanced pipeline loader with comprehensive fallback strategies"""
    
    @staticmethod
    def get_configuration_hierarchy(device: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Get prioritized configuration list for device"""
        if device == "NPU":
            return [
                ("reshape_npu", get_device_config("NPU", "reshape_npu")),
                ("enhanced_npu", get_device_config("NPU", "enhanced_npu")), 
                ("basic_npu", get_device_config("NPU", "basic")),
                ("minimal_npu", {}),
                ("cpu_fallback", get_device_config("CPU", "basic"))
            ]
        else:
            return [
                ("optimized_cpu", get_device_config("CPU", "enhanced")),
                ("basic_cpu", get_device_config("CPU", "basic")),
                ("minimal_cpu", {})
            ]
    
    @staticmethod
    def try_load_pipeline(model_path: str, device: str) -> Tuple[Optional[ov_genai.LLMPipeline], str, str]:
        """Try loading pipeline with comprehensive fallback strategy"""
        configs = PipelineLoader.get_configuration_hierarchy(device)
        
        for config_name, config in configs:
            try:
                target_device = "CPU" if "cpu" in config_name else device
                print(f"🔄 Attempting {target_device} with {config_name} configuration...")
                
                if config:
                    pipe = ov_genai.LLMPipeline(model_path, target_device, **config)
                else:
                    pipe = ov_genai.LLMPipeline(model_path, target_device)
                    
                print(f"✅ Success: {target_device} with {config_name}")
                return pipe, target_device, config_name
                
            except Exception as e:
                error_msg = str(e)
                if "compile" in error_msg.lower() or "npuw" in error_msg.lower():
                    print(f"⚠️  {config_name} failed (compilation): {error_msg[:100]}...")
                elif "memory" in error_msg.lower():
                    print(f"⚠️  {config_name} failed (memory): {error_msg[:100]}...")
                else:
                    print(f"⚠️  {config_name} failed: {error_msg[:100]}...")
                continue
        
        return None, device, "failed"

# Load pipeline with comprehensive fallback
try:
    pipe, device_used, config_used = PipelineLoader.try_load_pipeline(config.model_path, config.primary_device)
    
    if not pipe:
        raise RuntimeError("All configuration attempts failed")
    
    # Load and configure tokenizer with enhanced settings
    print("🔤 Loading tokenizer with enhanced configuration...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    # Enhanced tokenizer configuration
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Determine capabilities based on loaded configuration
    npu_optimized = ("npu" in config_used.lower() and device_used == "NPU")
    supports_reshaping = ("reshape" in config_used)
    
    print(f"✅ Model loaded: {device_used} with {config_used} configuration")
    print(f"🎯 NPU optimized: {npu_optimized}")
    print(f"📐 Supports reshaping: {supports_reshaping}")
    
except Exception as e:
    print(f"❌ Fatal error during model loading: {e}")
    print("💡 Try running with CPU device or check model path")
    exit(1)

load_end_time = time.time()
load_end_time = time.time()
load_time_seconds = load_end_time - load_start_time
print(f"⚡ Model loaded in {load_time_seconds:.2f} seconds")
print(f"🔋 Final device: {device_used}")
print(f"⚙️ Final configuration: {config_used}")
print(f"🏁 Ready for RAG-enhanced inference!")

# Store load time for performance tracking
performance_tracker.load_time = load_time_seconds

# --- Professional Streaming Class with Advanced Text Processing ---
class ProfessionalStreamer(ov_genai.StreamerBase):
    """Enhanced streamer with professional text processing and performance monitoring"""
    
    def __init__(self, tokenizer: AutoTokenizer, batch_size: int = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.q = queue.Queue()
        self.accumulated_tokens = []
        self.full_response = ""
        self.batch_size = batch_size or limits.STREAMING_BATCH_SIZE
        self.token_count = 0
        self.start_time = time.time()
        
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning with context-aware processing"""
        # Comprehensive special token removal
        special_tokens = [
            '<|im_start|>', '<|im_end|>', '<|system|>', '<|user|>', '<|assistant|>',
            '<|endoftext|>', '<|end|>', '<|start|>', '</s>', '<s>',
            '[INST]', '[/INST]', '<pad>', '<unk>', '<mask>',
            '\n\n\n', '\t\t', '  '  # Extra whitespace patterns
        ]
        
        cleaned = text
        for token in special_tokens:
            cleaned = cleaned.replace(token, '' if token.startswith('<') else ' ')
        
        # Advanced whitespace normalization while preserving structure
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)     # Normalize spaces/tabs
        cleaned = re.sub(r'\n ', '\n', cleaned)       # Remove space after newline
        
        return cleaned.strip()
    
    def put(self, token_id: int) -> bool:
        self.accumulated_tokens.append(token_id)
        self.token_count += 1
        
        # Process tokens in batches for smoother streaming
        if len(self.accumulated_tokens) % self.batch_size == 0 or self.token_count < 5:
            try:
                # Use skip_special_tokens for cleaner output
                decoded_text = self.tokenizer.decode(
                    self.accumulated_tokens, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            except Exception:
                # Fallback decoding
                decoded_text = self.tokenizer.decode(self.accumulated_tokens)
            
            if len(decoded_text) > len(self.full_response):
                new_text = decoded_text[len(self.full_response):]
                self.full_response = decoded_text
                
                if new_text.strip():  # Only send meaningful text
                    cleaned_new_text = self.clean_text(new_text)
                    if cleaned_new_text:
                        self.q.put(cleaned_new_text)
        
        return False
    
    def end(self):
        # Final cleanup and performance logging
        if self.full_response:
            final_cleaned = self.clean_text(self.full_response)
            if final_cleaned != self.full_response.strip():
                # Send final cleaned version if different
                self.q.put("\n[Final cleanup applied]")
        
        elapsed = time.time() - self.start_time
        tokens_per_sec = self.token_count / elapsed if elapsed > 0 else 0
        print(f"🚀 Streaming completed: {self.token_count} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
        
        self.q.put(None)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        item = self.q.get()
        if item is None:
            raise StopIteration
        return item

# --- Advanced Token Management with Context Awareness ---
class ContextAwareTokenManager:
    """Advanced token management with semantic awareness and NPU optimization"""
    
    @staticmethod
    def semantic_truncation(text: str, max_length: int, preserve_context: bool = True) -> str:
        """Advanced semantic-aware truncation with context preservation"""
        if len(text) <= max_length:
            return text
        
        # Strategy 1: Sentence boundary truncation
        sentences = re.split(r'[.!?]+\s+', text)
        if len(sentences) > 1:
            current_length = 0
            preserved_sentences = []
            
            for sentence in sentences:
                test_length = current_length + len(sentence) + 2
                if test_length <= max_length * 0.85:  # Leave buffer for context
                    preserved_sentences.append(sentence)
                    current_length = test_length
                else:
                    break
            
            if preserved_sentences and preserve_context:
                result = '. '.join(preserved_sentences) + '.'
                remaining = text[len(result):].strip()
                if len(remaining) > 30:
                    context_preview = remaining[:50].strip()
                    result += f" [...context: {context_preview}...]"
                return result
        
        # Strategy 2: Paragraph boundary truncation  
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1 and len(paragraphs[0]) < max_length * 0.7:
            return paragraphs[0] + f" [...{len(paragraphs)-1} more paragraphs...]"
        
        # Strategy 3: Smart word boundary with context
        truncated = text[:max_length - 100]  # Leave room for context
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.6:
            truncated = truncated[:last_space]
        
        # Add intelligent context preview
        remaining_preview = text[len(truncated):len(truncated) + 60].strip()
        return f"{truncated} [...continues: {remaining_preview}...]"
    
    @staticmethod
    def calculate_conversation_tokens(conversation: List[Dict], tokenizer) -> int:
        """Accurate token counting for conversation management"""
        if not conversation:
            return 0
            
        try:
            # Use chat template for accurate token counting
            formatted = tokenizer.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            return len(tokenizer.encode(formatted))
        except Exception as e:
            print(f"⚠️  Token calculation fallback: {e}")
            # Fallback: rough estimation
            total_chars = sum(len(msg.get('content', '')) for msg in conversation)
            return int(total_chars * 0.25)  # Rough char-to-token ratio

    @staticmethod
    def smart_truncate_conversation(
        conversation: List[Dict], 
        tokenizer, 
        max_tokens: int = None,
        preserve_system: bool = True
    ) -> List[Dict]:
        """Intelligent conversation truncation with context preservation"""
        if max_tokens is None:
            max_tokens = limits.CONVERSATION_LIMIT
        
        if len(conversation) <= 2:  # System + current message
            return conversation
        
        print(f"🔄 Smart conversation truncation (target: {max_tokens} tokens)")
        
        # Calculate initial token count
        initial_tokens = ContextAwareTokenManager.calculate_conversation_tokens(conversation, tokenizer)
        print(f"📊 Initial tokens: {initial_tokens}")
        
        if initial_tokens <= max_tokens:
            return conversation
        
        # Preserve system message and current user message
        system_msg = conversation[0] if preserve_system else None
        current_msg = conversation[-1] if conversation else None
        middle_conversation = conversation[1:-1] if len(conversation) > 2 else []
        
        # Smart removal strategy: remove oldest exchanges first
        while middle_conversation and initial_tokens > max_tokens:
            # Remove in pairs (user-assistant exchanges) when possible
            if len(middle_conversation) >= 2:
                # Find user-assistant pairs to remove together
                for i in range(len(middle_conversation) - 1):
                    if (middle_conversation[i].get('role') == 'user' and 
                        middle_conversation[i + 1].get('role') == 'assistant'):
                        removed_pair = [middle_conversation.pop(i), middle_conversation.pop(i)]
                        print(f"🗑️  Removed exchange: {len(removed_pair[0].get('content', ''))} + {len(removed_pair[1].get('content', ''))} chars")
                        break
                else:
                    # No pairs found, remove oldest single message
                    removed = middle_conversation.pop(0)
                    print(f"🗑️  Removed: {removed.get('role')} ({len(removed.get('content', ''))} chars)")
            else:
                # Remove remaining single message
                removed = middle_conversation.pop(0)
                print(f"🗑️  Removed: {removed.get('role')} ({len(removed.get('content', ''))} chars)")
            
            # Recalculate token count
            test_conversation = []
            if system_msg:
                test_conversation.append(system_msg)
            test_conversation.extend(middle_conversation)
            if current_msg:
                test_conversation.append(current_msg)
            
            initial_tokens = ContextAwareTokenManager.calculate_conversation_tokens(test_conversation, tokenizer)
            print(f"📊 Updated tokens: {initial_tokens}")
        
        # Reconstruct final conversation
        final_conversation = []
        if system_msg:
            final_conversation.append(system_msg)
        final_conversation.extend(middle_conversation)
        if current_msg:
            final_conversation.append(current_msg)
        
        return final_conversation

# --- RAG-Enhanced Chat Function with Comprehensive Features ---
def rag_enhanced_chat_function(message: str, history: list, generation_mode: str = "balanced"):
    """RAG-enhanced chat function with advanced features:
    - Stateful session management
    - Dynamic generation configuration
    - Context-aware token management  
    - Professional streaming with monitoring
    - Intelligent error handling
    """
    request_start = time.time()
    
    try:
        # Validate input
        if not message.strip():
            yield history
            return
        
        # Select generation configuration based on mode
        if generation_mode == "creative":
            current_config = GenerationConfigManager.get_creative_config()
        elif generation_mode == "precise":
            current_config = GenerationConfigManager.get_precise_config()
        else:
            current_config = GenerationConfigManager.get_balanced_config()
        
        print(f"🎛️  Using {generation_mode} mode (T={current_config.temperature})")
        
        # Enhanced message processing with semantic truncation
        if len(message) > limits.MAX_MESSAGE_LENGTH:
            original_length = len(message)
            message = ContextAwareTokenManager.semantic_truncation(message, limits.MAX_MESSAGE_LENGTH)
            print(f"📝 Message truncated: {original_length} → {len(message)} chars")
            
            # Provide user feedback about truncation
            history.append({
                "role": "assistant",
                "content": f"⚠️  Your message was truncated from {original_length:,} to {len(message)} characters for NPU optimization."
            })
            yield history
        
        # Initialize professional streaming
        streamer = ProfessionalStreamer(tokenizer)
        
        # Build conversation with system prompt
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        conversation.append({"role": "user", "content": message})
        
        # Apply intelligent conversation management
        max_tokens = limits.MAX_PROMPT_LEN if npu_optimized else limits.CONVERSATION_LIMIT
        conversation = ContextAwareTokenManager.smart_truncate_conversation(
            conversation, tokenizer, max_tokens=max_tokens
        )
        
        # Calculate final token usage
        final_tokens = ContextAwareTokenManager.calculate_conversation_tokens(conversation, tokenizer)
        print(f"🎯 Final conversation tokens: {final_tokens}")
        
        # Emergency token limit check
        if final_tokens > limits.EMERGENCY_LIMIT:
            error_msg = f"⚠️  Conversation too long ({final_tokens} tokens). Please start a new chat."
            history.append({"role": "assistant", "content": error_msg})
            yield history
            return
        
        # Update UI with user message
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        yield history
        
        # Check if we should use stateful chat API
        if config.use_stateful_chat and hasattr(pipe, 'generate'):
            # For stateful API, we only send the current message
            print("💬 Using stateful chat mode (sending current message only)")
            
            def generate_stateful():
                try:
                    pipe.generate(message, current_config, streamer)
                except Exception as e:
                    print(f"❌ Stateful generation error: {e}")
                    streamer.q.put(f"❌ Generation failed: {str(e)[:100]}...")
                    streamer.q.put(None)
        else:
            # Traditional full-prompt mode
            print("📜 Using full-prompt mode")
            prompt = tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            
            def generate_full():
                try:
                    pipe.generate(prompt, current_config, streamer)
                except Exception as e:
                    print(f"❌ Full-prompt generation error: {e}")
                    streamer.q.put(f"❌ Generation failed: {str(e)[:100]}...")
                    streamer.q.put(None)
        
        # Execute generation in separate thread
        generation_thread = Thread(target=generate_stateful if config.use_stateful_chat else generate_full)
        generation_thread.start()
        
        # Stream updates to UI with enhanced processing
        for chunk in streamer:
            if chunk:  # Only process non-empty chunks
                history[-1]["content"] += chunk
                yield history
        
        generation_thread.join(timeout=30.0)  # Prevent hanging
        
        # Update performance metrics
        request_time = time.time() - request_start
        output_tokens = len(tokenizer.encode(history[-1]["content"])) if history and history[-1]["content"] else 0
        
        performance_tracker.update(request_time, final_tokens, output_tokens, config_used)
        metrics = performance_tracker.get_metrics()
        
        print(f"🏁 Request completed: {request_time:.2f}s, {final_tokens}→{output_tokens} tokens ({metrics.get('tokens_per_second', 0):.1f} tok/s)")

    except Exception as e:
        error_msg = str(e)
        print(f"❌ RAG-enhanced chat error: {error_msg}")
        
        # Intelligent error handling with specific responses
        if "token" in error_msg.lower() or "length" in error_msg.lower():
            if "2048" in error_msg or "1024" in error_msg:
                error_response = f"⚠️  NPU token limit exceeded. Current conversation: {len(history)} messages. Please start a new chat or use a shorter message."
            else:
                error_response = "⚠️  Message too long for current NPU configuration. Try breaking it into smaller parts."
        elif "memory" in error_msg.lower() or "allocation" in error_msg.lower():
            error_response = "⚠️  NPU memory limit reached. Starting a new conversation will help."
        elif "compile" in error_msg.lower() or "npuw" in error_msg.lower():
            error_response = "⚠️  NPU compilation issue detected. The model may need to be re-optimized for your NPU configuration."
        elif "timeout" in error_msg.lower():
            error_response = "⏱️  Generation timed out. This may be due to high NPU load. Please try again."
        else:
            error_response = f"❌ Unexpected error: {error_msg[:150]}... Please try again or restart the chat."
        
        # Ensure we don't duplicate error messages
        if not history or history[-1].get("role") != "assistant":
            history.append({"role": "assistant", "content": error_response})
        else:
            history[-1]["content"] = error_response
        
        yield history

# --- Advanced Performance Monitoring ---
class EnhancedPerformanceTracker:
    """Comprehensive performance tracking with detailed metrics"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.total_requests = 0
        self.response_times = []
        self.input_tokens = []
        self.output_tokens = []
        self.configurations = []
        self.session_start = time.time()
        
    def update(self, response_time: float, input_tokens: int, output_tokens: int, config_type: str):
        """Update metrics with new request data"""
        self.total_requests += 1
        self.response_times.append(response_time)
        self.input_tokens.append(input_tokens)
        self.output_tokens.append(output_tokens)
        self.configurations.append(config_type)
        
        # Maintain sliding window
        if len(self.response_times) > self.window_size:
            self.response_times.pop(0)
            self.input_tokens.pop(0)
            self.output_tokens.pop(0)
            self.configurations.pop(0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.response_times:
            return {
                "total_requests": 0,
                "session_duration": "0s",
                "avg_response_time": 0.0,
                "tokens_per_second": 0.0,
                "current_config": "none"
            }
        
        session_duration = time.time() - self.session_start
        avg_response_time = np.mean(self.response_times)
        total_output_tokens = sum(self.output_tokens)
        total_time = sum(self.response_times)
        tokens_per_second = total_output_tokens / total_time if total_time > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "session_duration": f"{session_duration:.1f}s",
            "avg_response_time": round(avg_response_time, 2),
            "avg_input_tokens": round(np.mean(self.input_tokens), 1),
            "avg_output_tokens": round(np.mean(self.output_tokens), 1),
            "tokens_per_second": round(tokens_per_second, 1),
            "current_config": self.configurations[-1] if self.configurations else "unknown",
            "window_size": len(self.response_times),
            "device": device_used,
            "supports_reshaping": supports_reshaping
        }

# Initialize performance tracker
performance_tracker = EnhancedPerformanceTracker()

# --- RAG-Enhanced Gradio Interface ---
# Create RAG-enhanced interface with advanced controls
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="RAG-Enhanced Qwen Chat",
    css="""
    .gradio-container { max-width: 1400px; margin: auto; }
    .chatbot { height: 650px; }
    .control-panel { background: linear-gradient(90deg, #f0f9ff 0%, #e0f2fe 100%); padding: 15px; border-radius: 10px; margin: 10px 0; }
    .metrics-panel { background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; }
    .status-indicator { font-weight: bold; padding: 4px 8px; border-radius: 4px; }
    .status-success { background: #dcfce7; color: #166534; }
    .status-warning { background: #fef3c7; color: #92400e; }
    """,
    fill_height=True
) as demo:
    
    # Enhanced header with detailed status
    with gr.Row():
        gr.Markdown(f"""
        # 🤖 RAG-Enhanced Qwen Chat with Advanced NPU Optimization
        
        <div class="status-indicator status-success">**Device**: {device_used}</div> 
        <div class="status-indicator status-{'success' if npu_optimized else 'warning'}">**Config**: {config_used}</div>
        <div class="status-indicator status-success">**Model**: Qwen3-8B INT4-CW</div>
        
        **Features**: Stateful sessions • NPU reshaping • Context-aware truncation • Performance monitoring  
        **Token Limits**: {limits.MAX_PROMPT_LEN} prompt • {limits.CONVERSATION_LIMIT} conversation • Dynamic management
        """)
    
    # Main chat interface with enhanced features
    chatbot = gr.Chatbot(
        label="RAG-Enhanced Conversation", 
        height=650,
        type='messages',
        avatar_images=("👤", "🤖"),
        show_copy_button=True,
        show_share_button=False,
        bubble_full_width=False,
        render_markdown=True
    )
    
    # Enhanced input controls with generation mode selection
    with gr.Row():
        with gr.Column(scale=8):
            msg = gr.Textbox(
                label="Your Message", 
                placeholder=f"Ask me anything... (Auto-truncated at {limits.MAX_MESSAGE_LENGTH} chars for NPU efficiency)", 
                max_lines=4,
                lines=2
            )
        with gr.Column(scale=2):
            generation_mode = gr.Dropdown(
                label="Generation Mode",
                choices=["balanced", "creative", "precise"],
                value="balanced",
                info="Adjust creativity vs accuracy"
            )
            submit_btn = gr.Button("🚀 Send", variant="primary", size="lg")
    
    # Advanced control panel
    with gr.Row(elem_classes=["control-panel"]):
        with gr.Column(scale=1):
            clear_btn = gr.Button("🗑️ New Chat", variant="secondary")
            info_btn = gr.Button("ℹ️ System Info", variant="secondary")
        with gr.Column(scale=1):
            perf_btn = gr.Button("📊 Metrics", variant="secondary")
            export_btn = gr.Button("💾 Export Chat", variant="secondary")
        with gr.Column(scale=2):
            # Real-time status display
            status_display = gr.HTML(
                f"<div class='metrics-panel'>Ready • {config_used} mode • {device_used} device</div>"
            )
        
    # Collapsible advanced panels
    with gr.Row(visible=False) as metrics_row:
        with gr.Column(scale=1, elem_classes=["metrics-panel"]):
            gr.Markdown("### 📊 Performance Metrics")
            metrics_display = gr.JSON(label="Detailed Metrics", visible=True)
        with gr.Column(scale=1, elem_classes=["metrics-panel"]):
            gr.Markdown("### 🎯 Generation Settings")
            current_temp = gr.Number(label="Temperature", value=0.7, interactive=False)
            current_top_p = gr.Number(label="Top-p", value=0.9, interactive=False)
            current_tokens = gr.Number(label="Max Tokens", value=limits.MAX_PROMPT_LEN, interactive=False)
    
    with gr.Row(visible=False) as export_row:
        with gr.Column():
            gr.Markdown("### 💾 Export Options")
            export_format = gr.Dropdown(
                label="Format",
                choices=["JSON", "Markdown", "Plain Text"],
                value="Markdown"
            )
            export_output = gr.Textbox(
                label="Exported Chat",
                lines=10,
                max_lines=20
            )
            
    # Helper functions for UI interactions
    def update_generation_settings(mode: str):
        """Update generation settings display based on mode"""
        if mode == "creative":
            config = GenerationConfigManager.get_creative_config()
        elif mode == "precise":
            config = GenerationConfigManager.get_precise_config()
        else:
            config = GenerationConfigManager.get_balanced_config()
        
        return (
            gr.update(value=config.temperature),
            gr.update(value=config.top_p),
            gr.update(value=config.max_new_tokens)
        )
    
    def export_conversation(history: list, format_type: str) -> str:
        """Export conversation in specified format"""
        if not history:
            return "No conversation to export."
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        if format_type == "JSON":
            export_data = {
                "timestamp": timestamp,
                "device": device_used,
                "configuration": config_used,
                "conversation": history
            }
            return json.dumps(export_data, indent=2)
        
        elif format_type == "Markdown":
            lines = [f"# RAG-Enhanced Qwen Chat Export"]
            lines.append(f"**Date**: {timestamp}")
            lines.append(f"**Device**: {device_used}")
            lines.append(f"**Configuration**: {config_used}")
            lines.append("\n---\n")
            
            for msg in history:
                role = msg.get('role', 'unknown').title()
                content = msg.get('content', '')
                lines.append(f"## {role}\n")
                lines.append(f"{content}\n")
            
            return "\n".join(lines)
        
        else:  # Plain Text
            lines = [f"RAG-Enhanced Qwen Chat Export - {timestamp}"]
            lines.append(f"Device: {device_used} | Config: {config_used}")
            lines.append("=" * 50)
            
            for msg in history:
                role = msg.get('role', 'unknown').upper()
                content = msg.get('content', '')
                lines.append(f"\n[{role}]\n{content}")
            
            return "\n".join(lines)
    
    # Enhanced event handlers with advanced features
    def submit_with_mode(message: str, history: list, mode: str):
        """Submit message with specified generation mode"""
        return rag_enhanced_chat_function(message, history, mode)
    
    # Message submission with generation mode
    msg.submit(
        submit_with_mode, 
        [msg, chatbot, generation_mode], 
        chatbot
    ).then(
        lambda: gr.update(value=""), None, [msg], queue=False
    ).then(
        update_generation_settings,
        [generation_mode],
        [current_temp, current_top_p, current_tokens]
    )
    
    submit_btn.click(
        submit_with_mode, 
        [msg, chatbot, generation_mode], 
        chatbot
    ).then(
        lambda: gr.update(value=""), None, [msg], queue=False
    ).then(
        update_generation_settings,
        [generation_mode],
        [current_temp, current_top_p, current_tokens]
    )
    
    # Generation mode change updates settings display
    generation_mode.change(
        update_generation_settings,
        [generation_mode],
        [current_temp, current_top_p, current_tokens]
    )
    
    # Enhanced clear function with session management
    def clear_chat_session():
        """Clear chat and reset session if using stateful mode"""
        if config.use_stateful_chat and hasattr(pipe, 'finish_chat'):
            try:
                pipe.finish_chat()
                pipe.start_chat(SYSTEM_PROMPT)
                print("🔄 Stateful chat session reset")
            except Exception as e:
                print(f"⚠️  Session reset warning: {e}")
        return [], "", "Chat cleared • New session started"
    
    clear_btn.click(
        clear_chat_session,
        None, 
        [chatbot, msg, status_display],
        queue=False
    )
    
    def show_system_info():
        """Display comprehensive system information"""
        config_details = {
            "reshape_npu": "Full NPU reshaping with static shapes",
            "enhanced_npu": "Enhanced NPU with NPUW optimizations", 
            "basic_npu": "Basic NPU configuration",
            "minimal_npu": "Minimal NPU settings",
            "cpu_fallback": "CPU fallback mode"
        }
        
        info = f"""
        ## 🔧 System Configuration
        - **Device**: {device_used} ({"NPU-optimized" if npu_optimized else "Standard"})
        - **Configuration**: {config_details.get(config_used, config_used)}
        - **Model Path**: `{config.model_path}`
        - **Cache Directory**: `{config.cache_dir}`
        
        ## 🎯 Performance Specifications  
        - **Max Prompt Length**: {limits.MAX_PROMPT_LEN:,} tokens
        - **Conversation Limit**: {limits.CONVERSATION_LIMIT:,} tokens
        - **Message Length**: {limits.MAX_MESSAGE_LENGTH} characters
        - **Supports Reshaping**: {'Yes' if supports_reshaping else 'No'}
        
        ## 🧠 Model Architecture
        - **Base Model**: Qwen3ForCausalLM (8B parameters)
        - **Quantization**: INT4 channel-wise
        - **Optimization**: {'Stateful chat sessions' if config.use_stateful_chat else 'Traditional prompting'}
        - **Streaming**: Professional with batch processing
        
        ## 🚀 RAG Enhancements
        - NPU model reshaping with static shapes
        - Context-aware conversation management
        - Semantic truncation algorithms
        - Multi-tier configuration fallback
        - Real-time performance monitoring
        - Advanced generation controls
        """
        return info
    
    def toggle_metrics_display():
        return gr.update(visible=True)
    
    def toggle_export_display():
        return gr.update(visible=True)
    
    info_btn.click(
        lambda: gr.Info(show_system_info()),
        None, None
    )
    
    perf_btn.click(
        toggle_metrics_display,
        None, [metrics_row]
    ).then(
        lambda: performance_tracker.get_metrics(),
        None, metrics_display
    )
    
    export_btn.click(
        toggle_export_display,
        None, [export_row]
    )
    
    # Export functionality
    export_format.change(
        export_conversation,
        [chatbot, export_format],
        export_output
    )
    
    # Enhanced example prompts with generation mode suggestions
    gr.Examples(
        examples=[
            ["Explain quantum computing in simple terms", "precise"],
            ["Write a creative story about AI and humans", "creative"], 
            ["Write a Python function to calculate factorial", "precise"],
            ["What are some innovative ideas for renewable energy?", "creative"],
            ["How do neural networks work?", "balanced"],
            ["Compare NPU vs GPU for AI inference", "balanced"]
        ],
        inputs=[msg, generation_mode],
        label="Example Prompts (with suggested generation modes)"
    )

    # Initialize stateful chat session if enabled
    if config.use_stateful_chat and hasattr(pipe, 'start_chat'):
        try:
            pipe.start_chat(SYSTEM_PROMPT)
            print("💬 Stateful chat session initialized")
        except Exception as e:
            print(f"⚠️  Stateful chat initialization warning: {e}")

if __name__ == "__main__":
    print("\n🌐 Launching RAG-Enhanced Gradio Interface...")
    print(f"✨ Configuration: {device_used} with {config_used}")
    print("🚀 RAG-Enhanced Features Enabled:")
    print(f"   • NPU model reshaping: {'Yes' if supports_reshaping else 'No'}")
    print(f"   • Stateful chat sessions: {'Yes' if config.use_stateful_chat else 'No'}")
    print(f"   • Advanced generation controls: 3 modes available")
    print(f"   • Context-aware token management: {limits.MAX_PROMPT_LEN} token limit")
    print(f"   • Professional streaming with batch processing")
    print(f"   • Comprehensive error handling and fallback strategies")
    print(f"   • Real-time performance monitoring and metrics")
    print(f"   • Advanced UI with export and configuration controls")
    
    demo.queue(
        max_size=20,  # Increased queue size for better handling
        default_concurrency_limit=4
    ).launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        show_tips=True,
        inbrowser=True  # Auto-open browser
    )