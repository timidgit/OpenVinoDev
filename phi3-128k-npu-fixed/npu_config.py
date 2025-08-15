# NPU Configuration for Exported Model
# Generated automatically - use with your NPU pipeline

import openvino_genai as ov_genai
import openvino.properties as props
import openvino.properties.hint as hints

# NPU Pipeline-specific configuration (critical for compilation)
pipeline_config = {
    "MAX_PROMPT_LEN": 2048,        # Must match export static shape
    "MIN_RESPONSE_LEN": 256   # Minimum response tokens
}

# NPUW (NPU Wrapper) configuration for compilation success
npuw_config = {
    "NPU_USE_NPUW": "YES",                    # Enable NPU Wrapper
    "NPUW_LLM": "YES",                        # Enable LLM-specific optimizations
    "NPUW_LLM_BATCH_DIM": 0,                  # Batch dimension index
    "NPUW_LLM_SEQ_LEN_DIM": 1,               # Sequence dimension index
    "NPUW_LLM_MAX_PROMPT_LEN": 2048, # Must match MAX_PROMPT_LEN
    "NPUW_LLM_MIN_RESPONSE_LEN": 256, # Must match MIN_RESPONSE_LEN
}

# Device configuration for performance
device_config = {
    hints.performance_mode: hints.PerformanceMode.LATENCY,
    props.cache_dir: ".npu_cache"
}

def create_pipeline(model_path, device="NPU"):
    """Create optimized NPU pipeline with this exported model"""
    
    # Combine all configurations
    all_config = {**device_config, **pipeline_config, **npuw_config}
    
    try:
        print("Loading NPU-optimized model...")
        pipe = ov_genai.LLMPipeline(model_path, device, **all_config)
        print("✓ Successfully loaded model on NPU")
        return pipe
    except Exception as e:
        print(f"❌ Failed to load on NPU: {e}")
        print("Trying fallback configurations...")
        
        # Fallback: Try minimal NPUW config
        try:
            minimal_config = {**device_config, **npuw_config}
            pipe = ov_genai.LLMPipeline(model_path, device, **minimal_config)
            print("✓ Loaded with minimal NPUW configuration")
            return pipe
        except Exception as e2:
            print(f"❌ All NPU configurations failed: {e2}")
            return None

# Usage example:
# pipe = create_pipeline(r"phi3-128k-npu-fixed")
# response = pipe.generate("Hello, how are you?")
