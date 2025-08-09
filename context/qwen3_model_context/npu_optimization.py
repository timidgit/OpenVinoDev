# Qwen3 NPU Optimization Guide
# =============================
#
# PRIORITY: ⭐⭐⭐⭐⭐ (Critical for NPU deployment)
#
# This file contains comprehensive NPU optimization strategies specifically
# for Qwen3-8B model deployment with OpenVINO GenAI on Intel NPU hardware.
#
# Key Focus Areas:
# - NPUW (NPU Wrapper) configuration for successful compilation
# - Memory management for 8B parameter models
# - Performance optimization techniques
# - Troubleshooting NPU-specific issues

import os
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

# =======================================
# QWEN3 NPU CONFIGURATION PROFILES
# =======================================

@dataclass
class NPUPerformanceProfile:
    """NPU performance configuration profile"""
    max_prompt_len: int
    min_response_len: int
    cache_mode: str
    performance_hint: str
    compilation_strategy: str
    memory_optimization: str

# Pre-configured profiles for different use cases
QWEN3_NPU_PROFILES = {
    "conservative": NPUPerformanceProfile(
        max_prompt_len=1024,
        min_response_len=128,
        cache_mode="OPTIMIZE_SPEED",
        performance_hint="LATENCY",
        compilation_strategy="FAST",
        memory_optimization="HIGH"
    ),
    
    "balanced": NPUPerformanceProfile(
        max_prompt_len=2048,
        min_response_len=256,
        cache_mode="OPTIMIZE_SPEED", 
        performance_hint="LATENCY",
        compilation_strategy="BALANCED",
        memory_optimization="MEDIUM"
    ),
    
    "aggressive": NPUPerformanceProfile(
        max_prompt_len=4096,
        min_response_len=512,
        cache_mode="OPTIMIZE_SPEED",
        performance_hint="THROUGHPUT",
        compilation_strategy="OPTIMAL",
        memory_optimization="LOW"
    )
}

# =======================================
# QWEN3 NPUW CONFIGURATION BUILDER
# =======================================

class Qwen3NPUConfigBuilder:
    """Build optimized NPU configuration for Qwen3 models"""
    
    def __init__(self, profile: str = "balanced"):
        if profile not in QWEN3_NPU_PROFILES:
            raise ValueError(f"Unknown profile: {profile}. Available: {list(QWEN3_NPU_PROFILES.keys())}")
        
        self.profile = QWEN3_NPU_PROFILES[profile]
        self.config = {}
    
    def build_base_config(self) -> Dict[str, Any]:
        """Build base OpenVINO configuration"""
        # The generic 'PERFORMANCE_HINT' conflicts with the more specific NPUW hints.
        # By removing it, we allow the NPUW-specific hints (e.g., NPUW_LLM_GENERATE_HINT)
        # to be correctly applied by the OpenVINO plugin.
        return {
            # OpenVINO Core Properties
            "CACHE_DIR": ".ovcache_qwen3_npu",
            "NUM_STREAMS": 1,  # NPU works best with single stream
            "INFERENCE_NUM_THREADS": 1,
            
            # NPU-specific settings
            "DEVICE_PRIORITY": "NPU",
            "NPU_COMPILATION_MODE_PARAMS": self.profile.compilation_strategy,
        }
    
    def build_npuw_config(self) -> Dict[str, Any]:
        """Build NPUW (NPU Wrapper) configuration - REQUIRED for compilation"""
        return {
            # NPUW Core Settings (MANDATORY)
            "NPU_USE_NPUW": "YES",              # Enable NPU Wrapper
            "NPUW_LLM": "YES",                  # Enable LLM-specific optimizations
            "NPUW_LLM_BATCH_DIM": 0,            # Batch dimension index
            "NPUW_LLM_SEQ_LEN_DIM": 1,          # Sequence length dimension index
            
            # Qwen3-specific NPUW settings
            "NPUW_LLM_MAX_PROMPT_LEN": self.profile.max_prompt_len,
            "NPUW_LLM_MIN_RESPONSE_LEN": self.profile.min_response_len,
            
            # Performance hints for NPUW (must use BEST_PERF or FAST_COMPILE)
            "NPUW_LLM_PREFILL_HINT": "BEST_PERF",
            "NPUW_LLM_GENERATE_HINT": "BEST_PERF",
            
            # Advanced NPUW settings (removed unsupported options)
            "NPUW_WEIGHTS_BANK": "YES",         # Memory optimization
            "NPUW_FOLD_ELTWISE_UP": "YES"       # Optimization
        }
    
    def build_memory_config(self) -> Dict[str, Any]:
        """Build memory optimization configuration"""
        base_memory = {
            "CACHE_MODE": self.profile.cache_mode,
            "NPU_MEMORY_POOL_SIZE": "AUTO",
            "NPUW_CACHE_WEIGHTS": "YES",
        }
        
        # Add memory optimization based on profile
        if self.profile.memory_optimization == "HIGH":
            base_memory.update({
                "NPUW_LLM_MIN_RESPONSE_LEN": min(128, self.profile.min_response_len),
                "NPUW_COMPRESS_WEIGHTS": "YES",
                "NPU_LOW_MEMORY_MODE": "YES"
            })
        elif self.profile.memory_optimization == "MEDIUM":
            base_memory.update({
                "NPUW_COMPRESS_WEIGHTS": "AUTO",
            })
        
        return base_memory
    
    def build_complete_config(self, **overrides) -> Dict[str, Any]:
        """Build complete NPU configuration with all optimizations"""
        config = {}
        
        # Merge all configuration sections
        config.update(self.build_base_config())
        config.update(self.build_npuw_config()) 
        config.update(self.build_memory_config())
        
        # Apply user overrides
        config.update(overrides)
        
        return config

# =======================================
# QWEN3 NPU COMPILATION VALIDATOR
# =======================================

class Qwen3NPUCompilationValidator:
    """Validate NPU compilation requirements for Qwen3"""
    
    REQUIRED_NPUW_SETTINGS = [
        "NPU_USE_NPUW",
        "NPUW_LLM", 
        "NPUW_LLM_BATCH_DIM",
        "NPUW_LLM_SEQ_LEN_DIM",
        "NPUW_LLM_MAX_PROMPT_LEN",
        "NPUW_LLM_MIN_RESPONSE_LEN"
    ]
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate NPU configuration for compilation success
        
        Returns:
            (is_valid, missing_settings)
        """
        missing = []
        
        for setting in cls.REQUIRED_NPUW_SETTINGS:
            if setting not in config:
                missing.append(setting)
            elif config[setting] in [None, "", "NO", "False"]:
                missing.append(f"{setting} (should not be empty/NO/False)")
        
        # Validate specific values
        if "NPU_USE_NPUW" in config and config["NPU_USE_NPUW"] != "YES":
            missing.append("NPU_USE_NPUW must be 'YES'")
        
        if "NPUW_LLM" in config and config["NPUW_LLM"] != "YES":
            missing.append("NPUW_LLM must be 'YES'")
        
        # Validate prompt length constraints
        max_prompt = config.get("NPUW_LLM_MAX_PROMPT_LEN", 0)
        if isinstance(max_prompt, (int, str)) and int(max_prompt) > 4096:
            missing.append(f"NPUW_LLM_MAX_PROMPT_LEN ({max_prompt}) exceeds NPU memory limit (4096)")
        
        return len(missing) == 0, missing
    
    @classmethod
    def diagnose_compilation_failure(cls, error_message: str) -> Dict[str, Any]:
        """
        Diagnose common NPU compilation failures
        
        Returns:
            Dictionary with diagnosis and suggested fixes
        """
        diagnosis = {
            "error_type": "unknown",
            "likely_cause": "unknown",
            "suggested_fixes": []
        }
        
        error_lower = error_message.lower()

        # Add check for configuration parsing errors (most specific first)
        if "unsupported" in error_lower and ("option" in error_lower or "parse" in error_lower):
            diagnosis.update({
                "error_type": "configuration_parsing_error",
                "likely_cause": "An invalid value was provided for an NPU configuration key (e.g., NPUW_LLM_GENERATE_HINT).",
                "suggested_fixes": [
                    "Verify all NPUW hint values are supported (e.g., 'BEST_PERF' or 'FAST_COMPILE')",
                    "Ensure the generic 'PERFORMANCE_HINT' is removed when specific 'NPUW_LLM_*_HINT' keys are used",
                    "Check for typos in configuration keys",
                    "Remove conflicting generic and specific hint settings"
                ]
            })
            return diagnosis
        
        # Common error patterns
        if "failed to compile model0_fcew000__0" in error_lower:
            diagnosis.update({
                "error_type": "npuw_compilation_failure",
                "likely_cause": "Missing or incorrect NPUW configuration",
                "suggested_fixes": [
                    "Ensure NPU_USE_NPUW=YES is set",
                    "Verify NPUW_LLM=YES is configured", 
                    "Check NPUW_LLM_MAX_PROMPT_LEN matches pipeline config",
                    "Try conservative profile with smaller prompt length"
                ]
            })
        
        elif "memory" in error_lower or "allocation" in error_lower:
            diagnosis.update({
                "error_type": "memory_error",
                "likely_cause": "NPU memory constraints exceeded",
                "suggested_fixes": [
                    "Reduce NPUW_LLM_MAX_PROMPT_LEN to 1024 or lower",
                    "Enable NPU_LOW_MEMORY_MODE=YES",
                    "Use conservative profile",
                    "Consider CPU fallback for large contexts"
                ]
            })
        
        elif "device" in error_lower or "npu" in error_lower:
            diagnosis.update({
                "error_type": "device_error", 
                "likely_cause": "NPU device or driver issue",
                "suggested_fixes": [
                    "Verify NPU drivers are installed and up-to-date",
                    "Check if NPU is accessible via OpenVINO device list",
                    "Try restarting the application",
                    "Consider CPU fallback"
                ]
            })
        
        return diagnosis

# =======================================
# QWEN3 NPU PERFORMANCE MONITOR
# =======================================

class Qwen3NPUPerformanceMonitor:
    """Monitor NPU performance metrics for Qwen3"""
    
    def __init__(self):
        self.metrics = {
            "load_time": 0,
            "first_token_latency": 0,
            "tokens_per_second": 0,
            "compilation_time": 0,
            "memory_usage": 0,
            "cache_hit_rate": 0
        }
        
        self.load_start_time = None
        self.generation_start_time = None
        self.token_count = 0
    
    def start_load_timing(self):
        """Start timing model load"""
        self.load_start_time = time.time()
    
    def end_load_timing(self):
        """End timing model load"""
        if self.load_start_time:
            self.metrics["load_time"] = time.time() - self.load_start_time
            self.load_start_time = None
    
    def start_generation_timing(self):
        """Start timing generation"""
        self.generation_start_time = time.time()
        self.token_count = 0
    
    def record_token(self):
        """Record a token generation"""
        if self.generation_start_time:
            self.token_count += 1
            
            # Calculate first token latency
            if self.token_count == 1:
                self.metrics["first_token_latency"] = time.time() - self.generation_start_time
    
    def end_generation_timing(self):
        """End timing generation"""
        if self.generation_start_time and self.token_count > 0:
            total_time = time.time() - self.generation_start_time
            self.metrics["tokens_per_second"] = self.token_count / total_time
            self.generation_start_time = None
    
    def get_performance_report(self) -> str:
        """Generate performance report"""
        report = f"""
Qwen3 NPU Performance Report
============================
Load Time: {self.metrics['load_time']:.2f}s
First Token Latency: {self.metrics['first_token_latency']:.3f}s  
Tokens/Second: {self.metrics['tokens_per_second']:.1f}
Token Count: {self.token_count}

Performance Analysis:
- Load time {'✅ Good' if self.metrics['load_time'] < 90 else '⚠️ Slow'} (target: <90s)
- First token {'✅ Good' if self.metrics['first_token_latency'] < 2.0 else '⚠️ Slow'} (target: <2s)  
- Generation rate {'✅ Good' if self.metrics['tokens_per_second'] >= 15 else '⚠️ Slow'} (target: ≥15 tok/s)
        """
        return report.strip()

# =======================================
# QWEN3 NPU DEPLOYMENT UTILITIES
# =======================================

class Qwen3NPUDeployment:
    """Complete deployment utilities for Qwen3 on NPU"""
    
    def __init__(self, model_path: str, profile: str = "balanced"):
        self.model_path = model_path
        self.profile = profile
        self.config_builder = Qwen3NPUConfigBuilder(profile)
        self.performance_monitor = Qwen3NPUPerformanceMonitor()
        self.pipeline = None
    
    def deploy(self, **config_overrides) -> Optional[Any]:
        """
        Deploy Qwen3 model on NPU with comprehensive error handling
        
        Returns:
            Initialized LLMPipeline or None if deployment failed
        """
        import openvino_genai as ov_genai
        
        # Build configuration
        config = self.config_builder.build_complete_config(**config_overrides)
        
        # Validate configuration
        is_valid, missing = Qwen3NPUCompilationValidator.validate_config(config)
        if not is_valid:
            print(f"❌ Invalid NPU configuration. Missing: {missing}")
            return None
        
        print(f"🚀 Deploying Qwen3 on NPU with {self.profile} profile...")
        print(f"📊 Max prompt length: {config['NPUW_LLM_MAX_PROMPT_LEN']}")
        print(f"📊 Min response length: {config['NPUW_LLM_MIN_RESPONSE_LEN']}")
        
        # Attempt deployment with performance monitoring
        self.performance_monitor.start_load_timing()
        
        try:
            self.pipeline = ov_genai.LLMPipeline(self.model_path, "NPU", **config)
            self.performance_monitor.end_load_timing()
            
            print(f"✅ Qwen3 deployed successfully on NPU")
            print(f"⏱️ Load time: {self.performance_monitor.metrics['load_time']:.1f}s")
            
            return self.pipeline
            
        except Exception as e:
            self.performance_monitor.end_load_timing()
            
            print(f"❌ NPU deployment failed: {str(e)}")
            
            # Diagnose the failure
            diagnosis = Qwen3NPUCompilationValidator.diagnose_compilation_failure(str(e))
            print(f"🔍 Diagnosis: {diagnosis['likely_cause']}")
            
            for i, fix in enumerate(diagnosis['suggested_fixes'], 1):
                print(f"   {i}. {fix}")
            
            return None
    
    def benchmark(self, test_prompts: List[str] = None) -> Dict[str, Any]:
        """Run performance benchmark on deployed model"""
        
        if not self.pipeline:
            return {"error": "No pipeline deployed"}
        
        if not test_prompts:
            test_prompts = [
                "Explain quantum computing in simple terms.",
                "Write a Python function to calculate fibonacci numbers.", 
                "What are the benefits of using Intel NPU for AI inference?"
            ]
        
        results = []
        
        print("🔬 Running Qwen3 NPU benchmark...")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"Test {i}/{len(test_prompts)}: {prompt[:50]}...")
            
            # Configure generation
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = 256
            config.do_sample = False  # Greedy for consistent timing
            
            # Run test with timing
            self.performance_monitor.start_generation_timing()
            
            try:
                self.pipeline.start_chat()
                response = self.pipeline.generate(prompt, config)
                self.pipeline.finish_chat()
                
                # Record results
                self.performance_monitor.end_generation_timing()
                
                results.append({
                    "prompt": prompt,
                    "response_length": len(response.split()),
                    "first_token_latency": self.performance_monitor.metrics["first_token_latency"],
                    "tokens_per_second": self.performance_monitor.metrics["tokens_per_second"]
                })
                
            except Exception as e:
                results.append({
                    "prompt": prompt,
                    "error": str(e)
                })
        
        # Calculate averages
        successful_tests = [r for r in results if "error" not in r]
        if successful_tests:
            avg_latency = sum(r["first_token_latency"] for r in successful_tests) / len(successful_tests)
            avg_throughput = sum(r["tokens_per_second"] for r in successful_tests) / len(successful_tests)
            
            benchmark_summary = {
                "profile": self.profile,
                "total_tests": len(test_prompts),
                "successful_tests": len(successful_tests),
                "average_first_token_latency": avg_latency,
                "average_tokens_per_second": avg_throughput,
                "load_time": self.performance_monitor.metrics["load_time"],
                "detailed_results": results
            }
        else:
            benchmark_summary = {
                "profile": self.profile, 
                "error": "All tests failed",
                "detailed_results": results
            }
        
        return benchmark_summary
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on performance"""
        
        recommendations = []
        metrics = self.performance_monitor.metrics
        
        # Load time recommendations
        if metrics["load_time"] > 120:
            recommendations.append("Consider using a smaller model or different quantization")
            recommendations.append("Ensure NPU drivers are optimized")
        elif metrics["load_time"] > 90:
            recommendations.append("Load time is acceptable but could be improved with driver updates")
        
        # Generation speed recommendations  
        if metrics["tokens_per_second"] < 10:
            recommendations.append("Very slow generation - check NPU utilization and memory")
            recommendations.append("Consider switching to conservative profile")
        elif metrics["tokens_per_second"] < 15:
            recommendations.append("Generation speed could be improved - try balanced profile")
        
        # First token latency
        if metrics["first_token_latency"] > 3.0:
            recommendations.append("High first token latency - reduce max prompt length")
            recommendations.append("Consider greedy decoding for faster response start")
        
        if not recommendations:
            recommendations.append("Performance looks good! Current configuration is well-optimized.")
        
        return recommendations

# =======================================
# USAGE EXAMPLES
# =======================================

def example_basic_npu_deployment():
    """Example of basic NPU deployment"""
    
    model_path = "C:\\OpenVinoModels\\qwen3-8b-int4-cw-ov"
    
    # Deploy with balanced profile
    deployment = Qwen3NPUDeployment(model_path, profile="balanced")
    pipeline = deployment.deploy()
    
    if pipeline:
        print("✅ Deployment successful!")
        
        # Run benchmark
        benchmark_results = deployment.benchmark()
        print("\n📊 Benchmark Results:")
        for key, value in benchmark_results.items():
            if key != "detailed_results":
                print(f"  {key}: {value}")
    
        # Get recommendations
        recommendations = deployment.get_optimization_recommendations()
        print("\n💡 Optimization Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")

def example_custom_npu_configuration():
    """Example of custom NPU configuration"""
    
    # Build custom config
    builder = Qwen3NPUConfigBuilder("aggressive")
    
    # Add custom overrides
    custom_config = builder.build_complete_config(
        NPUW_LLM_MAX_PROMPT_LEN=8192,  # Larger context
        NPUW_COMPRESS_WEIGHTS="YES",   # Extra compression
        NPU_LOW_MEMORY_MODE="YES"      # Memory optimization
    )
    
    print("Custom NPU Configuration:")
    for key, value in custom_config.items():
        if key.startswith(("NPU", "NPUW")):
            print(f"  {key}: {value}")

def example_configuration_validation():
    """Example of configuration validation"""
    
    # Test configuration
    test_config = {
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES", 
        "NPUW_LLM_BATCH_DIM": 0,
        "NPUW_LLM_SEQ_LEN_DIM": 1,
        "NPUW_LLM_MAX_PROMPT_LEN": 2048,
        "NPUW_LLM_MIN_RESPONSE_LEN": 256
    }
    
    # Validate
    is_valid, missing = Qwen3NPUCompilationValidator.validate_config(test_config)
    
    if is_valid:
        print("✅ Configuration is valid for NPU compilation")
    else:
        print("❌ Configuration issues found:")
        for issue in missing:
            print(f"  • {issue}")

if __name__ == "__main__":
    print("Qwen3 NPU Optimization Guide Loaded")
    print("=" * 50)
    
    # Show available profiles
    print("Available NPU Profiles:")
    for name, profile in QWEN3_NPU_PROFILES.items():
        print(f"  {name}: max_prompt={profile.max_prompt_len}, min_response={profile.min_response_len}")
    
    # Run validation example
    print("\nRunning configuration validation example...")
    example_configuration_validation()