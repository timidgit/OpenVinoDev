#!/usr/bin/env python3
"""
NPU-Compatible Qwen Model Export Script
=====================================

This script exports Qwen models with NPU-specific optimizations:
- Static shapes for KV-cache (required for NPU)
- Symmetric INT4 quantization
- Stateful KV-cache implementation
- Channel-wise quantization for >1B parameter models

Usage:
    python export_qwen_for_npu.py --model Qwen/Qwen2.5-7B-Instruct --output qwen2.5-7b-npu
"""

import argparse
import os
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import optimum.intel
        import openvino
        import transformers
        print("✓ All required packages found")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Install with: pip install optimum-intel transformers")
        return False
    return True

def export_qwen_for_npu(model_name, output_dir, max_seq_len=2048, min_response_len=256):
    """
    Export Qwen model optimized for NPU inference
    
    Args:
        model_name: Hugging Face model name (e.g., "Qwen/Qwen2.5-7B-Instruct") 
        output_dir: Directory to save the exported model
        max_seq_len: Maximum sequence length for static shapes
        min_response_len: Minimum response length for NPU optimization
    """
    
    print(f"🚀 Exporting {model_name} for NPU...")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚙️  Max sequence length: {max_seq_len}")
    print(f"⚙️  Min response length: {min_response_len}")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # NPU Export Command - Critical parameters for NPU compatibility
    cmd = [
        "optimum-cli", "export", "openvino",
        "--model", model_name,
        "--task", "text-generation-with-past",  # Enable KV-cache
        "--weight-format", "int4",              # NPU-optimized quantization
        "--sym",                                # Symmetric quantization (NPU preferred)
        "--group-size", "-1",                   # Channel-wise for >1B models
        "--ratio", "1.0",                       # Full model quantization
        "--trust-remote-code",                  # For Qwen models
        output_dir
    ]
    
    print("\n🔄 Running export command:")
    print(" ".join(cmd))
    print("\n" + "="*60)
    
    # Execute the export
    import subprocess
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Export completed successfully!")
        print(result.stdout)
        
        # Create NPU configuration file
        create_npu_config_file(output_dir, max_seq_len, min_response_len)
        
        print(f"\n🎉 NPU-compatible model exported to: {output_dir}")
        print(f"📋 Configuration saved to: {output_dir}/npu_config.py")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Export failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    return True

def create_npu_config_file(output_dir, max_seq_len, min_response_len):
    """Create a configuration file with NPU-specific settings"""
    
    config_content = f'''# NPU Configuration for Exported Model
# Generated automatically - use with your NPU pipeline

import openvino_genai as ov_genai
import openvino.properties as props
import openvino.properties.hint as hints

# NPU Pipeline-specific configuration (critical for compilation)
pipeline_config = {{
    "MAX_PROMPT_LEN": {max_seq_len},        # Must match export static shape
    "MIN_RESPONSE_LEN": {min_response_len}   # Minimum response tokens
}}

# NPUW (NPU Wrapper) configuration for compilation success
npuw_config = {{
    "NPU_USE_NPUW": "YES",                    # Enable NPU Wrapper
    "NPUW_LLM": "YES",                        # Enable LLM-specific optimizations
    "NPUW_LLM_BATCH_DIM": 0,                  # Batch dimension index
    "NPUW_LLM_SEQ_LEN_DIM": 1,               # Sequence dimension index
    "NPUW_LLM_MAX_PROMPT_LEN": {max_seq_len}, # Must match MAX_PROMPT_LEN
    "NPUW_LLM_MIN_RESPONSE_LEN": {min_response_len}, # Must match MIN_RESPONSE_LEN
}}

# Device configuration for performance
device_config = {{
    hints.performance_mode: hints.PerformanceMode.LATENCY,
    props.cache_dir: ".npu_cache"
}}

def create_pipeline(model_path, device="NPU"):
    """Create optimized NPU pipeline with this exported model"""
    
    # Combine all configurations
    all_config = {{**device_config, **pipeline_config, **npuw_config}}
    
    try:
        print("Loading NPU-optimized model...")
        pipe = ov_genai.LLMPipeline(model_path, device, **all_config)
        print("✓ Successfully loaded model on NPU")
        return pipe
    except Exception as e:
        print(f"❌ Failed to load on NPU: {{e}}")
        print("Trying fallback configurations...")
        
        # Fallback: Try minimal NPUW config
        try:
            minimal_config = {{**device_config, **npuw_config}}
            pipe = ov_genai.LLMPipeline(model_path, device, **minimal_config)
            print("✓ Loaded with minimal NPUW configuration")
            return pipe
        except Exception as e2:
            print(f"❌ All NPU configurations failed: {{e2}}")
            return None

# Usage example:
# pipe = create_pipeline(r"{output_dir}")
# response = pipe.generate("Hello, how are you?")
'''
    
    config_path = Path(output_dir) / "npu_config.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"📄 Created NPU configuration: {config_path}")

def main():
    parser = argparse.ArgumentParser(description="Export Qwen models for NPU inference")
    parser.add_argument("--model", "-m", 
                       default="Qwen/Qwen2.5-7B-Instruct",
                       help="Hugging Face model name")
    parser.add_argument("--output", "-o", 
                       default="qwen-npu-model",
                       help="Output directory")
    parser.add_argument("--max-seq-len", 
                       type=int, default=2048,
                       help="Maximum sequence length (static shape)")
    parser.add_argument("--min-response", 
                       type=int, default=256,
                       help="Minimum response length for NPU")
    
    args = parser.parse_args()
    
    print("🔧 NPU Model Export Tool")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Export the model
    success = export_qwen_for_npu(
        args.model, 
        args.output,
        args.max_seq_len,
        args.min_response
    )
    
    if success:
        print("\n✅ Export completed successfully!")
        print(f"🚀 Model ready for NPU inference: {args.output}")
        print("\n📖 Next steps:")
        print(f"1. Test with: python -c \"from {args.output}.npu_config import create_pipeline; create_pipeline(r'{args.output}')\"")
        print(f"2. Update your Gradio script to use: {args.output}")
    else:
        print("\n❌ Export failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()