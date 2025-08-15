"""
Model Deployment and System Initialization
=========================================

Handles OpenVINO GenAI pipeline deployment with comprehensive error handling,
NPU optimization, and system validation.
"""

import os
import time
from typing import Any, Tuple, List
from typing_extensions import Literal

import openvino_genai as ov_genai
from transformers import AutoTokenizer

from .config import get_config

# Try to import OpenVINO properties with fallback
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

# Import Phi-3 NPU optimization patterns
try:
    from .npu_patterns import (
        get_npu_config_balanced,
        get_npu_config_conservative, 
        get_npu_config_aggressive,
        initialize_phi3_pipeline,
        PHI3_SPECIAL_TOKENS
    )
    ENHANCED_CONTEXT_AVAILABLE = True
    print("✅ Enhanced Phi-3 NPU context loaded successfully")
except ImportError as e:
    print(f"⚠️ Enhanced Phi-3 context not available: {e}")
    print("📝 Using fallback patterns - consider updating context path")
    ENHANCED_CONTEXT_AVAILABLE = False


# Type definitions
DeviceType = Literal["NPU", "CPU", "GPU", "AUTO"]
ProfileType = Literal["conservative", "balanced", "aggressive"]
ConfigDict = dict[str, Any]


class LLMConfigurationManager:
    """Advanced configuration management with Phi-3 optimization"""
    
    def __init__(self, profile: ProfileType = "balanced") -> None:
        """
        Initialize configuration manager with specified profile.
        
        Args:
            profile: NPU optimization profile (conservative, balanced, aggressive)
        """
        self.profile = profile
        self.enhanced_available = ENHANCED_CONTEXT_AVAILABLE
    
    def get_npu_config(self) -> ConfigDict:
        """
        Get complete NPU configuration with NPUW optimization.
        
        Returns:
            Dictionary containing NPU-specific configuration parameters
        """
        if ENHANCED_CONTEXT_AVAILABLE:
            # Use enhanced Phi-3-specific configuration
            if self.profile == "conservative":
                return get_npu_config_conservative()
            elif self.profile == "balanced":
                return get_npu_config_balanced()
            elif self.profile == "aggressive":
                return get_npu_config_aggressive()
            else:
                return get_npu_config_balanced()  # Default fallback
        else:
            # Fallback configuration optimized for Phi-3 128k context
            # Critical: Use correct NPUW hints to prevent compilation errors
            config = {
                "NPU_USE_NPUW": "YES",
                "NPUW_LLM": "YES", 
                "NPUW_LLM_BATCH_DIM": 0,
                "NPUW_LLM_SEQ_LEN_DIM": 1,
                "NPUW_LLM_MAX_PROMPT_LEN": 8192,  # Increased for Phi-3 128k context
                "NPUW_LLM_MIN_RESPONSE_LEN": 512,  # Increased for better responses
                "CACHE_MODE": "OPTIMIZE_SPEED",
                "NPUW_LLM_PREFILL_HINT": "FAST_COMPILE",  # Corrected: FAST_COMPILE for stable compilation
                "NPUW_LLM_GENERATE_HINT": "BEST_PERF"     # Corrected: BEST_PERF for optimal generation
            }
            
            # Add OpenVINO properties if available (no generic PERFORMANCE_HINT for NPU)
            if OPENVINO_PROPERTIES_AVAILABLE:
                cache_dir = get_config().get("deployment", "cache_directory", "./cache/.ovcache_phi3")
                config.update({
                    props.cache_dir: cache_dir
                })
            else:
                config.update({
                    "CACHE_DIR": get_config().get("deployment", "cache_directory", "./cache/.ovcache_phi3")
                })
            
            return config
    
    def get_cpu_config(self) -> ConfigDict:
        """
        Get optimized CPU configuration.
        
        Returns:
            Dictionary containing CPU-specific configuration parameters
        """
        if ENHANCED_CONTEXT_AVAILABLE:
            # Use Phi-3 optimized CPU configuration
            return {
                "MAX_PROMPT_LEN": 32768,  # Use Phi-3's 128k context capability
                "MIN_RESPONSE_LEN": 512,
                "CACHE_DIR": get_config().get("deployment", "cache_directory", "./cache/.ovcache_phi3") + "_cpu"
            }
        else:
            config = {
                "MAX_PROMPT_LEN": 16384,  # Much larger context for Phi-3 on CPU
                "MIN_RESPONSE_LEN": 512
            }
            
            # Add OpenVINO properties if available
            if OPENVINO_PROPERTIES_AVAILABLE:
                cache_dir = get_config().get("deployment", "cache_directory", "./cache/.ovcache_phi3") + "_cpu"
                config.update({
                    hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                    props.cache_dir: cache_dir,
                    props.streams.num: 2,
                    props.inference_num_threads: 0  # Auto-detect
                })
            else:
                config.update({
                    "PERFORMANCE_HINT": "THROUGHPUT",
                    "CACHE_DIR": get_config().get("deployment", "cache_directory", "./cache/.ovcache_phi3") + "_cpu",
                    "NUM_STREAMS": 2,
                    "INFERENCE_NUM_THREADS": 0  # Auto-detect
                })
            
            return config


def deploy_llm_pipeline(
    model_path: str, 
    target_device: DeviceType, 
    profile: ProfileType = "balanced"
) -> Tuple[Any, str, str, float]:
    """
    Deploy language model pipeline (Phi-3) with comprehensive error handling and optimization.
    
    Args:
        model_path: Path to the Phi-3 OpenVINO model directory
        target_device: Target device for deployment (NPU, CPU, GPU, AUTO)
        profile: NPU optimization profile
        
    Returns:
        Tuple of (pipeline, device_used, config_used, load_time)
        
    Raises:
        RuntimeError: If all deployment configurations fail
    """
    load_start_time = time.time()
    
    if ENHANCED_CONTEXT_AVAILABLE:
        print(f"🚀 Deploying Phi-3 with enhanced NPU context (profile: {profile})")
        
        # Use enhanced Phi-3 deployment with NPU patterns
        pipeline = initialize_phi3_pipeline(model_path, target_device, profile)
        
        if pipeline:
            load_time = time.time() - load_start_time
            return pipeline, target_device, f"enhanced_{profile}", load_time
        else:
            print("⚠️ Enhanced deployment failed, falling back to manual configuration")
    
    # Fallback to manual configuration
    print(f"🔄 Using manual pipeline deployment (target: {target_device})")
    
    config_manager = LLMConfigurationManager(profile)
    
    configurations = []
    
    # Create basic configurations with compatibility handling
    cache_dir = get_config().get("deployment", "cache_directory", "./cache/.ovcache_phi3")
    
    if OPENVINO_PROPERTIES_AVAILABLE:
        basic_npu_config = {hints.performance_mode: hints.PerformanceMode.LATENCY, props.cache_dir: cache_dir}
        basic_cpu_config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT, props.cache_dir: cache_dir}
    else:
        basic_npu_config = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": cache_dir}
        basic_cpu_config = {"PERFORMANCE_HINT": "THROUGHPUT", "CACHE_DIR": cache_dir}
    
    if target_device == "NPU":
        configurations = [
            ("enhanced_npu_phi3", target_device, config_manager.get_npu_config()),
            ("basic_npu", target_device, basic_npu_config),
            ("minimal_npu", target_device, {}),
            ("cpu_fallback", "CPU", config_manager.get_cpu_config())
        ]
    else:
        configurations = [
            ("optimized_cpu_phi3", target_device, config_manager.get_cpu_config()),
            ("basic_cpu", target_device, basic_cpu_config),
            ("minimal_cpu", target_device, {})
        ]
    
    for config_name, device, config in configurations:
        try:
            print(f"🔄 Trying {device} with {config_name} configuration...")
            
            if ENHANCED_CONTEXT_AVAILABLE:
                # Use enhanced Phi-3 initialization if available
                pipeline = initialize_phi3_pipeline(model_path, device, profile, **config)
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
            continue
    
    raise RuntimeError("All configurations failed. Check model path, device drivers, and NPUW configuration.")


def validate_system_requirements() -> List[str]:
    """Validate system requirements and return list of issues."""
    issues = []
    
    config = get_config()
    model_path = config.get("model", "path")
    target_device = config.get("deployment", "target_device")
    cache_dir = config.get("deployment", "cache_directory")
    
    # Check model path
    if not os.path.exists(model_path):
        issues.append(f"Model path does not exist: {model_path}")
    elif not os.path.isdir(model_path):
        issues.append(f"Model path is not a directory: {model_path}")
    else:
        # Check for required OpenVINO files
        required_files = ['openvino_model.xml', 'openvino_model.bin']
        for file_name in required_files:
            if not os.path.exists(os.path.join(model_path, file_name)):
                issues.append(f"Missing OpenVINO model file: {file_name}")
    
    # Check cache directory
    cache_parent = os.path.dirname(cache_dir)
    if not os.path.exists(cache_parent):
        try:
            os.makedirs(cache_parent, exist_ok=True)
        except PermissionError:
            issues.append(f"Cannot create cache directory: {cache_parent} (permission denied)")
        except Exception as e:
            issues.append(f"Cannot create cache directory: {cache_parent} ({str(e)})")
    
    # Check OpenVINO installation
    try:
        import openvino as ov
        core = ov.Core()
        available_devices = core.available_devices
        if target_device not in available_devices and target_device != "AUTO":
            issues.append(f"Target device '{target_device}' not available. Available: {available_devices}")
    except Exception as e:
        issues.append(f"OpenVINO not properly installed: {str(e)}")
    
    return issues


def initialize_system_with_validation():
    """Initialize system with comprehensive validation and error handling."""
    config = get_config()
    
    print("🔍 Validating system requirements...")
    issues = validate_system_requirements()
    
    if issues:
        print("❌ System validation failed:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\n🔧 Suggested fixes:")
        print("   • Set MODEL_PATH environment variable to correct model location")
        print("   • Install OpenVINO with: pip install openvino")
        print("   • For NPU: Install Intel NPU drivers from official site")
        print("   • Ensure model is in OpenVINO format (.xml/.bin files)")
        raise SystemExit(1)
    
    try:
        print("🚀 Initializing Enhanced Phi-3 Chat System...")
        
        # Get configuration values
        model_path = config.get("model", "path")
        target_device = config.get("deployment", "target_device", "NPU")
        npu_profile = config.get("deployment", "npu_profile", "balanced")
        
        print(f"📂 Model: {model_path}")
        print(f"🎯 Target Device: {target_device}")
        print(f"📊 Optimization Profile: {npu_profile}")
        print(f"🔧 Enhanced Context: {'Available' if ENHANCED_CONTEXT_AVAILABLE else 'Fallback Mode'}")
        
        # Deploy pipeline with comprehensive error handling
        pipeline, device_used, config_used, load_time = deploy_llm_pipeline(
            model_path, target_device, npu_profile
        )
        
        # Initialize tokenizer with error handling
        print("📚 Loading Phi-3 tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Configure tokenizer for Phi-3
            if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                
        except Exception as tokenizer_error:
            print(f"⚠️ Tokenizer loading failed: {tokenizer_error}")
            print("🔄 Attempting fallback tokenizer initialization...")
            try:
                # Fallback: try without trust_remote_code
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
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
        print(f"   Model Path: {model_path}")
        print(f"   Tokenizer: {tokenizer.__class__.__name__}")
        if ENHANCED_CONTEXT_AVAILABLE:
            print(f"   Special Tokens: {len(PHI3_SPECIAL_TOKENS)} Phi-3 special tokens available")
        print("=" * 60)
        
        return pipeline, tokenizer, device_used, config_used, load_time
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        print("\n🔧 Detailed diagnostics:")
        print(f"   Model Path: {config.get('model', 'path')}")
        print(f"   Target Device: {config.get('deployment', 'target_device')}")
        print(f"   Cache Directory: {config.get('deployment', 'cache_directory')}")
        print(f"   Enhanced Context: {ENHANCED_CONTEXT_AVAILABLE}")
        
        # Provide specific guidance based on error type
        error_str = str(e).lower()
        if "compile" in error_str:
            print("\n💡 NPU Compilation Error - Try:")
            print("   • Verify NPU drivers are installed")
            print("   • Check NPUW configuration compatibility")
            print("   • Try CPU fallback with: --device CPU")
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