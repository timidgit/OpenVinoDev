# Qwen3 Model Architecture & Configuration Guide
# =============================================
#
# PRIORITY: ⭐⭐⭐⭐⭐ (Critical for Qwen3 optimization)
#
# This file documents the Qwen3-8B model architecture, configuration patterns,
# and OpenVINO GenAI optimization strategies specific to this model.
#
# Based on analysis of: C:\OpenVinoModels\qwen3-8b-int4-cw-ov\config.json

# =======================================
# MODEL ARCHITECTURE SPECIFICATIONS
# =======================================

QWEN3_8B_ARCHITECTURE = {
    "model_type": "qwen3",
    "architecture_class": "Qwen3ForCausalLM",
    "parameters": "8B",
    
    # Core Architecture
    "hidden_size": 4096,
    "intermediate_size": 12288,  # Feed-forward network size
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,    # GQA (Grouped Query Attention)
    "head_dim": 128,             # hidden_size / num_attention_heads
    
    # Position & Context
    "max_position_embeddings": 40960,  # 40K context length
    "max_window_layers": 36,
    "rope_theta": 1000000,
    "sliding_window": None,
    "use_sliding_window": False,
    
    # Vocabulary & Tokens
    "vocab_size": 151936,
    "bos_token_id": 151643,     # <|endoftext|>
    "eos_token_id": 151645,     # <|im_end|>
    "pad_token_id": None,       # Uses bos_token for padding
    
    # Technical Details
    "attention_dropout": 0.0,
    "hidden_act": "silu",       # Swish activation
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-06,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": False,
    "use_cache": True
}

# =======================================
# QWEN3 OPTIMIZATION FOR OPENVINO GENAI
# =======================================

class Qwen3OpenVINOConfig:
    """Optimized OpenVINO configuration for Qwen3-8B models"""
    
    @staticmethod
    def get_npu_config():
        """NPU-specific configuration for Qwen3"""
        return {
            # Base OpenVINO properties (removed PERFORMANCE_HINT to avoid conflicts)
            "CACHE_DIR": ".ovcache_qwen3",
            "NUM_STREAMS": 1,
            "INFERENCE_NUM_THREADS": 1,
            
            # NPUW configuration (required for NPU compilation)
            "NPU_USE_NPUW": "YES",
            "NPUW_LLM": "YES",
            "NPUW_LLM_BATCH_DIM": 0,
            "NPUW_LLM_SEQ_LEN_DIM": 1,
            
            # Qwen3-specific NPUW settings
            "NPUW_LLM_MAX_PROMPT_LEN": 2048,    # Conservative for NPU
            "NPUW_LLM_MIN_RESPONSE_LEN": 256,
            "NPUW_LLM_PREFILL_HINT": "BEST_PERF",
            "NPUW_LLM_GENERATE_HINT": "BEST_PERF",
            
            # Memory optimization for 8B model
            "CACHE_MODE": "OPTIMIZE_SPEED",
            "NPUW_LLM_DYNAMIC_SHAPE": "NO"     # Better NPU compatibility
        }
    
    @staticmethod
    def get_cpu_config():
        """CPU-optimized configuration for Qwen3"""
        return {
            "PERFORMANCE_HINT": "THROUGHPUT",
            "CACHE_DIR": ".ovcache_qwen3_cpu", 
            "NUM_STREAMS": 2,
            "INFERENCE_NUM_THREADS": 0,        # Auto-detect
            
            # CPU-specific optimizations for 8B model
            "ENABLE_CPU_PINNING": True,
            "CPU_BIND_THREAD": "NUMA",
            
            # Larger prompt support on CPU
            "MAX_PROMPT_LEN": 8192,            # Use more of 40K context
            "MIN_RESPONSE_LEN": 512
        }
    
    @staticmethod 
    def get_generation_config():
        """Default generation configuration for Qwen3"""
        return {
            # From generation_config.json analysis
            "do_sample": True,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
            
            # Token IDs (critical for proper generation)
            "bos_token_id": 151643,  # <|endoftext|>
            "eos_token_id": [151645, 151643],  # <|im_end|>, <|endoftext|>
            "pad_token_id": 151643,  # Same as bos
            
            # Response controls
            "max_new_tokens": 2048,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            
            # Qwen3 specific
            "do_sample": True,       # Enable sampling for creative responses
            "early_stopping": True
        }

# =======================================
# QWEN3 PIPELINE INITIALIZATION PATTERNS
# =======================================

def initialize_qwen3_pipeline(model_path, device="NPU", **kwargs):
    """
    Robust Qwen3 pipeline initialization with proper error handling
    
    Args:
        model_path: Path to Qwen3 OpenVINO model
        device: Target device ("NPU", "CPU", "GPU")
        **kwargs: Additional configuration overrides
    
    Returns:
        Initialized LLMPipeline optimized for Qwen3
    """
    import openvino_genai as ov_genai
    from openvino import properties as props, hints
    
    config_provider = Qwen3OpenVINOConfig()
    
    # Get device-specific configuration
    if device == "NPU":
        base_config = config_provider.get_npu_config()
    elif device == "CPU":
        base_config = config_provider.get_cpu_config()
    elif device == "GPU":
        base_config = {
            "PERFORMANCE_HINT": "LATENCY",
            "CACHE_DIR": ".ovcache_qwen3_gpu",
            "GPU_ENABLE_LOOP_UNROLLING": False  # Better for large models
        }
    else:
        raise ValueError(f"Unsupported device: {device}")
    
    # Merge with user overrides
    final_config = {**base_config, **kwargs}
    
    # Three-tier initialization strategy for maximum compatibility
    try:
        # Tier 1: Full configuration with all optimizations
        print(f"Initializing Qwen3 on {device} with full optimization...")
        pipe = ov_genai.LLMPipeline(model_path, device, **final_config)
        print(f"✅ Qwen3 initialization successful with full config")
        return pipe
        
    except Exception as e1:
        print(f"⚠️ Full config failed: {e1}")
        
        try:
            # Tier 2: Basic device properties only
            basic_config = {
                "PERFORMANCE_HINT": "LATENCY",
                "CACHE_DIR": f".ovcache_qwen3_{device.lower()}"
            }
            pipe = ov_genai.LLMPipeline(model_path, device, **basic_config)
            print(f"✅ Qwen3 initialization successful with basic config")
            return pipe
            
        except Exception as e2:
            print(f"⚠️ Basic config failed: {e2}")
            
            # Tier 3: Minimal configuration
            pipe = ov_genai.LLMPipeline(model_path, device)
            print(f"✅ Qwen3 initialization successful with minimal config")
            return pipe

# =======================================
# QWEN3 PERFORMANCE CHARACTERISTICS
# =======================================

QWEN3_PERFORMANCE_PROFILES = {
    "NPU": {
        "typical_load_time": "60-75 seconds",
        "tokens_per_second": "15-25 tokens/sec",
        "max_concurrent_users": 1,
        "recommended_prompt_length": "< 2048 tokens",
        "memory_usage": "Low (NPU optimized)",
        
        # NPU-specific considerations
        "compilation_time": "High (first run)",
        "cache_benefits": "Significant",
        "batch_size": 1,  # NPU works best with single requests
        
        "optimization_notes": [
            "NPUW configuration is mandatory for compilation",
            "Conservative prompt lengths prevent memory issues",
            "Cache warmup improves subsequent loads significantly",
            "Greedy decoding often faster than sampling on NPU"
        ]
    },
    
    "CPU": {
        "typical_load_time": "10-20 seconds", 
        "tokens_per_second": "8-15 tokens/sec",
        "max_concurrent_users": 2,
        "recommended_prompt_length": "< 8192 tokens",
        "memory_usage": "High (8B parameters in RAM)",
        
        # CPU-specific considerations
        "compilation_time": "Low",
        "cache_benefits": "Moderate",
        "batch_size": 1,  # Sequential processing
        
        "optimization_notes": [
            "Larger context windows possible vs NPU",
            "Thread count auto-detection usually optimal", 
            "Memory usage scales with context length",
            "Good fallback when NPU unavailable"
        ]
    },
    
    "GPU": {
        "typical_load_time": "15-30 seconds",
        "tokens_per_second": "20-40 tokens/sec", 
        "max_concurrent_users": 1,
        "recommended_prompt_length": "< 4096 tokens",
        "memory_usage": "Very High (GPU VRAM required)",
        
        "optimization_notes": [
            "Requires adequate GPU VRAM (>8GB recommended)",
            "Fastest generation once loaded",
            "Good for high-throughput scenarios"
        ]
    }
}

# =======================================
# QWEN3 TROUBLESHOOTING GUIDE  
# =======================================

QWEN3_COMMON_ISSUES = {
    "compilation_failures": {
        "symptoms": ["Failed to compile Model0_FCEW000__0", "NPU compilation error"],
        "solutions": [
            "Ensure NPU_USE_NPUW=YES and NPUW_LLM=YES are set",
            "Check NPUW_LLM_MAX_PROMPT_LEN matches your pipeline config",
            "Verify model is in correct INT4 format for NPU",
            "Try reducing MAX_PROMPT_LEN to 1024 for problematic models"
        ]
    },
    
    "memory_issues": {
        "symptoms": ["Out of memory", "Allocation failed", "NPU memory exceeded"],
        "solutions": [
            "Reduce MAX_PROMPT_LEN and MIN_RESPONSE_LEN",
            "Use smaller batch sizes (batch_size=1)",
            "Enable CACHE_MODE=OPTIMIZE_SPEED",
            "Consider CPU fallback for long conversations"
        ]
    },
    
    "slow_generation": {
        "symptoms": ["Very slow token generation", "High latency"],
        "solutions": [
            "Verify NPU drivers are properly installed",
            "Check if model cache exists (.ovcache directory)",
            "Use PERFORMANCE_HINT=LATENCY",
            "Consider greedy decoding (temperature=0) for speed"
        ]
    },
    
    "tokenization_errors": {
        "symptoms": ["Special token errors", "Chat template issues"],
        "solutions": [
            "Check special token handling in your tokenizer",
            "Verify chat template is being applied correctly", 
            "Filter special tokens before display (see special_tokens.py)",
            "Ensure proper eos_token_id configuration"
        ]
    }
}

# =======================================
# USAGE EXAMPLES
# =======================================

def qwen3_basic_usage():
    """Basic Qwen3 usage example"""
    
    # Initialize pipeline
    model_path = "C:\\OpenVinoModels\\qwen3-8b-int4-cw-ov"
    pipe = initialize_qwen3_pipeline(model_path, device="NPU")
    
    # Configure generation
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 512
    config.temperature = 0.7
    config.do_sample = True
    
    # Generate response  
    pipe.start_chat()
    response = pipe.generate("Explain quantum computing", config)
    pipe.finish_chat()
    
    return response

def qwen3_streaming_usage():
    """Qwen3 streaming usage example"""
    
    class Qwen3Streamer(ov_genai.StreamerBase):
        def __init__(self, tokenizer):
            super().__init__()
            self.tokenizer = tokenizer
            self.tokens = []
        
        def put(self, token_id):
            self.tokens.append(token_id)
            # Filter Qwen3 special tokens
            if token_id not in [151644, 151645, 151646]:  # <|im_start|>, <|im_end|>, etc.
                decoded = self.tokenizer.decode([token_id])
                print(decoded, end='', flush=True)
            return False
        
        def end(self):
            print()  # New line at end
    
    # Usage
    model_path = "C:\\OpenVinoModels\\qwen3-8b-int4-cw-ov"
    pipe = initialize_qwen3_pipeline(model_path, device="NPU")
    
    # Note: tokenizer initialization would be needed
    # streamer = Qwen3Streamer(tokenizer)
    # pipe.start_chat()
    # pipe.generate("Hello!", config, streamer)
    # pipe.finish_chat()

# =======================================
# INTEGRATION WITH GRADIO
# =======================================

def create_qwen3_gradio_interface():
    """Create optimized Gradio interface for Qwen3"""
    
    import gradio as gr
    
    # Initialize Qwen3 pipeline globally
    global qwen3_pipe
    model_path = "C:\\OpenVinoModels\\qwen3-8b-int4-cw-ov"
    qwen3_pipe = initialize_qwen3_pipeline(model_path, device="NPU")
    
    def qwen3_chat_response(message, history):
        """Chat response function optimized for Qwen3"""
        
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 512
        config.temperature = 0.6  # Qwen3 default
        config.top_p = 0.95       # Qwen3 default
        config.do_sample = True
        
        # Qwen3 session management
        qwen3_pipe.start_chat()
        try:
            response = qwen3_pipe.generate(message, config)
            return response
        finally:
            qwen3_pipe.finish_chat()
    
    # Create interface with Qwen3-optimized settings
    interface = gr.ChatInterface(
        qwen3_chat_response,
        title="🎯 Qwen3-8B Chat (OpenVINO NPU)",
        description="Optimized for Intel NPU with Qwen3-specific configurations",
        examples=[
            "Explain the benefits of Intel NPU acceleration",
            "What are the key features of the Qwen3 model?",
            "How does OpenVINO optimize transformer models?"
        ],
        theme=gr.themes.Soft(),
        css=".container { max-width: 900px; margin: auto; }"
    )
    
    return interface

if __name__ == "__main__":
    # Example usage
    print("Qwen3 Model Configuration Loaded")
    print(f"Architecture: {QWEN3_8B_ARCHITECTURE['architecture_class']}")
    print(f"Parameters: {QWEN3_8B_ARCHITECTURE['parameters']}")
    print(f"Context Length: {QWEN3_8B_ARCHITECTURE['max_position_embeddings']:,}")
    print(f"Vocab Size: {QWEN3_8B_ARCHITECTURE['vocab_size']:,}")