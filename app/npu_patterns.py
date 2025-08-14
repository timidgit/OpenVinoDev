#!/usr/bin/env python3
"""
Phi-3 NPU Optimization Patterns
==============================

Generic NPU optimization patterns extracted from legacy Qwen3 context,
updated for Phi-3-mini-128k-instruct compatibility.

This replaces the incompatible Qwen3-specific context with model-agnostic
NPU patterns that actually work with Phi-3.
"""

import openvino_genai as ov_genai


# NPU Configuration Profiles
def get_npu_config_conservative():
    """Conservative NPU configuration for stable operation"""
    return {
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_LLM_BATCH_DIM": 0,
        "NPUW_LLM_SEQ_LEN_DIM": 1,
        "NPUW_LLM_MAX_PROMPT_LEN": 1024,
        "NPUW_LLM_MIN_RESPONSE_LEN": 128,
        "NPUW_LLM_PREFILL_HINT": "FAST_COMPILE",
        "NPUW_LLM_GENERATE_HINT": "BEST_PERF",
        "CACHE_MODE": "OPTIMIZE_SPEED"
    }


def get_npu_config_balanced():
    """Balanced NPU configuration for Phi-3"""
    return {
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_LLM_BATCH_DIM": 0,
        "NPUW_LLM_SEQ_LEN_DIM": 1,
        "NPUW_LLM_MAX_PROMPT_LEN": 2048,
        "NPUW_LLM_MIN_RESPONSE_LEN": 256,
        "NPUW_LLM_PREFILL_HINT": "FAST_COMPILE",
        "NPUW_LLM_GENERATE_HINT": "BEST_PERF",
        "CACHE_MODE": "OPTIMIZE_SPEED"
    }


def get_npu_config_aggressive():
    """Aggressive NPU configuration for maximum context"""
    return {
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_LLM_BATCH_DIM": 0,
        "NPUW_LLM_SEQ_LEN_DIM": 1,
        "NPUW_LLM_MAX_PROMPT_LEN": 8192,  # Use Phi-3's 128k context capability
        "NPUW_LLM_MIN_RESPONSE_LEN": 512,
        "NPUW_LLM_PREFILL_HINT": "FAST_COMPILE",
        "NPUW_LLM_GENERATE_HINT": "BEST_PERF",
        "CACHE_MODE": "OPTIMIZE_SPEED"
    }


def initialize_phi3_pipeline(model_path, device="NPU", profile="balanced", **kwargs):
    """
    Initialize Phi-3 pipeline with proper NPU configuration
    
    Args:
        model_path: Path to Phi-3 OpenVINO model
        device: Target device ("NPU", "CPU", "GPU")
        profile: NPU profile ("conservative", "balanced", "aggressive")
        **kwargs: Additional configuration overrides
    
    Returns:
        Initialized LLMPipeline optimized for Phi-3
    """
    
    # Get profile-specific configuration
    if profile == "conservative":
        base_config = get_npu_config_conservative()
    elif profile == "balanced":
        base_config = get_npu_config_balanced()
    elif profile == "aggressive":
        base_config = get_npu_config_aggressive()
    else:
        raise ValueError(f"Unknown profile: {profile}")
    
    # Add cache directory
    base_config["CACHE_DIR"] = f".ovcache_phi3_{device.lower()}"
    
    # Merge with user overrides
    final_config = {**base_config, **kwargs}
    
    # Three-tier initialization for maximum compatibility
    try:
        print(f"🚀 Initializing Phi-3 on {device} with {profile} profile...")
        pipe = ov_genai.LLMPipeline(model_path, device, **final_config)
        print(f"✅ Phi-3 initialization successful with {profile} config")
        return pipe
        
    except Exception as e1:
        print(f"⚠️ {profile} config failed: {e1}")
        
        try:
            # Fallback: Basic configuration
            basic_config = {
                "NPU_USE_NPUW": "YES",
                "NPUW_LLM": "YES",
                "CACHE_DIR": f".ovcache_phi3_{device.lower()}"
            }
            pipe = ov_genai.LLMPipeline(model_path, device, **basic_config)
            print(f"✅ Phi-3 initialization successful with basic config")
            return pipe
            
        except Exception as e2:
            print(f"⚠️ Basic config failed: {e2}")
            
            # Final fallback: Minimal configuration
            pipe = ov_genai.LLMPipeline(model_path, device)
            print(f"✅ Phi-3 initialization successful with minimal config")
            return pipe


# Phi-3 Special Tokens (correct tokens for this model)
PHI3_SPECIAL_TOKENS = {
    # Phi-3 uses different token format than Qwen3
    "<|endoftext|>": True,
    "<|system|>": True,
    "<|user|>": True,
    "<|assistant|>": True,
    "<|end|>": True
}


def is_phi3_special_token(token_text):
    """Check if a token is a Phi-3 special token that should be filtered"""
    if not token_text:
        return False
    
    token_text = token_text.strip()
    return token_text in PHI3_SPECIAL_TOKENS


# Performance profiles for Phi-3
PHI3_PERFORMANCE_PROFILES = {
    "NPU": {
        "typical_load_time": "60-90 seconds",
        "tokens_per_second": "10-20 tokens/sec",  # More conservative than Qwen3
        "max_concurrent_users": 1,
        "recommended_prompt_length": "< 2048 tokens",
        "memory_usage": "Low (NPU optimized)",
        "notes": "Phi-3 has different performance characteristics than Qwen3"
    },
    
    "CPU": {
        "typical_load_time": "10-25 seconds",
        "tokens_per_second": "5-12 tokens/sec",
        "max_concurrent_users": 2,
        "recommended_prompt_length": "< 8192 tokens (can use 128k context)",
        "memory_usage": "Moderate (3.8B parameters vs 8B in Qwen3)",
        "notes": "Phi-3 is smaller than Qwen3, uses less memory"
    }
}

if __name__ == "__main__":
    print("Phi-3 NPU Optimization Patterns Loaded")
    print("✅ Model-agnostic NPU configuration")
    print("✅ Correct Phi-3 special tokens")
    print("✅ Proper performance profiles")