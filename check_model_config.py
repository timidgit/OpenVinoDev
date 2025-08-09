#!/usr/bin/env python3
"""
Check current model configuration and identify export settings
"""

import json
from pathlib import Path

def check_model_config(model_path):
    """Check the configuration of an existing OpenVINO model"""
    
    model_dir = Path(model_path)
    config_file = model_dir / "config.json"
    
    print(f"🔍 Checking model: {model_path}")
    print("=" * 50)
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_path}")
        return
    
    if not config_file.exists():
        print(f"❌ config.json not found in {model_path}")
        return
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print("📋 Model Configuration:")
        print(f"   Model type: {config.get('model_type', 'Unknown')}")
        print(f"   Architecture: {config.get('architectures', 'Unknown')}")
        
        if 'max_position_embeddings' in config:
            print(f"   Max position embeddings: {config['max_position_embeddings']}")
        
        if 'hidden_size' in config:
            print(f"   Hidden size: {config['hidden_size']}")
            
        if 'num_attention_heads' in config:
            print(f"   Attention heads: {config['num_attention_heads']}")
            
        if 'vocab_size' in config:
            print(f"   Vocabulary size: {config['vocab_size']}")
        
        # Check for quantization info
        if 'quantization_config' in config:
            quant_config = config['quantization_config']
            print(f"\n🔧 Quantization Configuration:")
            for key, value in quant_config.items():
                print(f"   {key}: {value}")
        else:
            print("\n❓ No quantization configuration found")
        
        # Check OpenVINO files
        print(f"\n📁 OpenVINO Files:")
        xml_file = model_dir / "openvino_model.xml"
        bin_file = model_dir / "openvino_model.bin"
        
        if xml_file.exists():
            print(f"   ✓ Model XML: {xml_file.name} ({xml_file.stat().st_size / (1024*1024):.1f} MB)")
        else:
            print(f"   ❌ Missing: openvino_model.xml")
            
        if bin_file.exists():
            print(f"   ✓ Model weights: {bin_file.name} ({bin_file.stat().st_size / (1024*1024):.1f} MB)")
        else:
            print(f"   ❌ Missing: openvino_model.bin")
        
        # Check for tokenizer
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
        print(f"\n🔤 Tokenizer Files:")
        for tok_file in tokenizer_files:
            tok_path = model_dir / tok_file
            if tok_path.exists():
                print(f"   ✓ {tok_file}")
            else:
                print(f"   ❌ Missing: {tok_file}")
        
        # Detect likely source model
        print(f"\n🎯 Likely Source Model:")
        model_type = config.get('model_type', '').lower()
        if 'qwen' in model_type:
            print(f"   This appears to be a Qwen model")
            if config.get('hidden_size') == 4096:
                print(f"   Likely: Qwen2.5-7B or similar")
            elif config.get('hidden_size') == 3584:
                print(f"   Likely: Qwen2.5-3B or similar") 
        
        # Check if model supports NPU
        print(f"\n🖥️  NPU Compatibility Analysis:")
        print(f"   Current export likely has dynamic shapes (causing NPU compilation failure)")
        print(f"   Recommendation: Re-export with static shapes using export_qwen_for_npu.py")
        
        return config
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")

if __name__ == "__main__":
    model_path = r"C:\OpenVinoModels\qwen3-8b-int4-cw-ov"
    check_model_config(model_path)