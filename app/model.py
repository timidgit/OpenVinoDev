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

# Import enhanced context patterns
import sys
context_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "context")
sys.path.insert(0, context_path)

# Import Qwen3-specific optimizations
try:
    from qwen3_model_context.npu_optimization import (
        Qwen3NPUConfigBuilder, 
        Qwen3NPUDeployment,
        QWEN3_NPU_PROFILES
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


# Type definitions
DeviceType = Literal["NPU", "CPU", "GPU", "AUTO"]
ProfileType = Literal["conservative", "balanced", "aggressive"]
ConfigDict = dict[str, Any]


class Qwen3ConfigurationManager:
    """Advanced configuration management with Qwen3 optimization"""
    
    def __init__(self, profile: ProfileType = "balanced") -> None:
        """
        Initialize configuration manager with specified profile.
        
        Args:
            profile: NPU optimization profile (conservative, balanced, aggressive)
        """
        self.profile = profile
        self.config_builder: Any = None
        
        if ENHANCED_CONTEXT_AVAILABLE:
            self.config_builder = Qwen3NPUConfigBuilder(profile)
    
    def get_npu_config(self) -> ConfigDict:
        """
        Get complete NPU configuration with NPUW optimization.
        
        Returns:
            Dictionary containing NPU-specific configuration parameters
        """
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
                "NPUW_LLM_PREFILL_HINT": "LATENCY",
                "NPUW_LLM_GENERATE_HINT": "LATENCY"
            }
            
            # Add OpenVINO properties if available
            if OPENVINO_PROPERTIES_AVAILABLE:
                cache_dir = get_config().get("deployment", "cache_directory", "./cache/.ovcache_qwen3")
                config.update({
                    hints.performance_mode: hints.PerformanceMode.LATENCY,
                    props.cache_dir: cache_dir
                })
            else:
                config.update({
                    "PERFORMANCE_HINT": "LATENCY",
                    "CACHE_DIR": get_config().get("deployment", "cache_directory", "./cache/.ovcache_qwen3")
                })
            
            return config
    
    def get_cpu_config(self) -> ConfigDict:
        """
        Get optimized CPU configuration.
        
        Returns:
            Dictionary containing CPU-specific configuration parameters
        """
        if ENHANCED_CONTEXT_AVAILABLE and self.config_builder:
            return self.config_builder.build_complete_config()
        else:
            config = {
                "MAX_PROMPT_LEN": 4096,  # Larger context on CPU
                "MIN_RESPONSE_LEN": 512
            }
            
            # Add OpenVINO properties if available
            if OPENVINO_PROPERTIES_AVAILABLE:
                cache_dir = get_config().get("deployment", "cache_directory", "./cache/.ovcache_qwen3") + "_cpu"
                config.update({
                    hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                    props.cache_dir: cache_dir,
                    props.streams.num: 2,
                    props.inference_num_threads: 0  # Auto-detect
                })
            else:
                config.update({
                    "PERFORMANCE_HINT": "THROUGHPUT",
                    "CACHE_DIR": get_config().get("deployment", "cache_directory", "./cache/.ovcache_qwen3") + "_cpu",
                    "NUM_STREAMS": 2,
                    "INFERENCE_NUM_THREADS": 0  # Auto-detect
                })
            
            return config


def deploy_qwen3_pipeline(
    model_path: str, 
    target_device: DeviceType, 
    profile: ProfileType = "balanced"
) -> Tuple[Any, str, str, float]:
    """
    Deploy Qwen3 pipeline with comprehensive error handling and optimization.
    
    Args:
        model_path: Path to the Qwen3 OpenVINO model directory
        target_device: Target device for deployment (NPU, CPU, GPU, AUTO)
        profile: NPU optimization profile
        
    Returns:
        Tuple of (pipeline, device_used, config_used, load_time)
        
    Raises:
        RuntimeError: If all deployment configurations fail
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
    cache_dir = get_config().get("deployment", "cache_directory", "./cache/.ovcache_qwen3")
    
    if OPENVINO_PROPERTIES_AVAILABLE:
        basic_npu_config = {hints.performance_mode: hints.PerformanceMode.LATENCY, props.cache_dir: cache_dir}
        basic_cpu_config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT, props.cache_dir: cache_dir}
    else:
        basic_npu_config = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": cache_dir}
        basic_cpu_config = {"PERFORMANCE_HINT": "THROUGHPUT", "CACHE_DIR": cache_dir}
    
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
        print("   • Set QWEN3_MODEL_PATH environment variable to correct model location")
        print("   • Install OpenVINO with: pip install openvino")
        print("   • For NPU: Install Intel NPU drivers from official site")
        print("   • Ensure model is in OpenVINO format (.xml/.bin files)")
        raise SystemExit(1)
    
    try:
        print("🚀 Initializing Enhanced Qwen3 Chat System...")
        
        # Get configuration values
        model_path = config.get("model", "path")
        target_device = config.get("deployment", "target_device", "NPU")
        npu_profile = config.get("deployment", "npu_profile", "balanced")
        
        print(f"📂 Model: {model_path}")
        print(f"🎯 Target Device: {target_device}")
        print(f"📊 Optimization Profile: {npu_profile}")
        print(f"🔧 Enhanced Context: {'Available' if ENHANCED_CONTEXT_AVAILABLE else 'Fallback Mode'}")
        
        # Deploy pipeline with comprehensive error handling
        pipeline, device_used, config_used, load_time = deploy_qwen3_pipeline(
            model_path, target_device, npu_profile
        )
        
        # Initialize tokenizer with error handling
        print("📚 Loading Qwen3 tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Configure tokenizer for Qwen3
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
            from qwen3_model_context.special_tokens import QWEN3_SPECIAL_TOKENS
            print(f"   Special Tokens: {len(QWEN3_SPECIAL_TOKENS)} Qwen3 tokens loaded")
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