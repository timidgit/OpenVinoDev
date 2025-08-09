# Enhanced Qwen3 OpenVINO GenAI Chat Application

A production-ready chat application showcasing Qwen3-8B model deployment with Intel OpenVINO GenAI on NPU hardware. This project demonstrates advanced optimization techniques, professional error handling, and modern UI design patterns.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![OpenVINO](https://img.shields.io/badge/OpenVINO-2024.4+-orange.svg)

## ✨ Features

### 🚀 **Performance Optimization**
- **Complete Intel NPU optimization** with NPUW (NPU Wrapper) configuration
- **Qwen3-specific tuning** with model architecture awareness  
- **Intelligent device fallback** (NPU → CPU) with appropriate configurations
- **Professional performance monitoring** with real-time metrics

### 🎯 **Model Support**
- **Qwen3-8B INT4** optimized deployment
- **26+ special token filtering** for clean output
- **Chat template support** including tool calling and reasoning modes
- **Stateful conversation management** using OpenVINO GenAI best practices

### 🖥️ **User Experience**
- **Modern Gradio interface** with professional theming
- **Real-time streaming responses** with token-level filtering
- **Smart message processing** with intelligent truncation
- **Comprehensive error handling** with user-friendly diagnostics

### 🔧 **Developer Features**
- **Modular architecture** with clear separation of concerns
- **Comprehensive logging** and performance metrics
- **Robust fallback strategies** ensuring maximum compatibility
- **Professional error diagnostics** for troubleshooting

## 🛠️ Installation

### Prerequisites
- **Python 3.8+**
- **OpenVINO 2024.4+** with GenAI support
- **Intel NPU drivers** (for NPU acceleration)
- **Qwen3-8B INT4 model** in OpenVINO format

### Quick Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd OpenVinoDev
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up OpenVINO environment**
```bash
# Source OpenVINO environment variables
source /opt/intel/openvino/setupvars.sh  # Linux
# or
"C:\Program Files (x86)\Intel\openvino\setupvars.bat"  # Windows
```

4. **Configure the application**

**Option A: Environment Variables**
```bash
export QWEN3_MODEL_PATH="/path/to/your/qwen3-8b-int4-ov"
export TARGET_DEVICE="NPU"  # or "CPU"
export NPU_PROFILE="balanced"  # conservative, balanced, aggressive
```

**Option B: Configuration File (Recommended)**
```bash
# Copy example configuration
cp config.example.json config.json

# Edit config.json with your settings
nano config.json
```

5. **Run the application**
```bash
python gradio_qwen_enhanced.py
```

## 📋 Requirements

### Hardware
- **Recommended**: Intel NPU-enabled system (Meteor Lake, Arrow Lake, or later)
- **Alternative**: Intel CPU with AVX-512 support
- **Memory**: 8GB+ RAM for optimal performance

### Software
```
openvino-genai>=2024.4
gradio>=4.0.0
transformers>=4.30.0
numpy>=1.21.0
```

See `requirements.txt` for complete dependencies.

## 🚀 Usage

### Basic Usage
```python
python gradio_qwen_enhanced.py
```

The application will:
1. **Auto-detect** available hardware (NPU/CPU)
2. **Deploy** Qwen3 model with optimal configuration
3. **Launch** web interface at `http://127.0.0.1:7860`

### Configuration Options

**Environment Variables:**
```bash
export QWEN3_MODEL_PATH="/path/to/model"      # Model location
export CACHE_DIR="/path/to/cache"             # Cache directory
export NPU_PROFILE="balanced"                 # balanced|conservative|aggressive
export MAX_MESSAGE_LENGTH="400"               # Max input length
```

**NPU Profiles:**
- **Conservative**: Lower memory usage, smaller contexts
- **Balanced**: Optimal for most use cases (default)
- **Aggressive**: Maximum performance, higher memory usage

## 🏗️ Architecture

### Core Components

```
src/
├── gradio_qwen_enhanced.py     # Main application
├── context/                    # Enhanced context system
│   ├── qwen3_model_context/    # Qwen3-specific optimizations
│   ├── gradio_patterns/        # UI patterns
│   └── gradio_testing/         # Testing utilities
└── legacy/                     # Previous versions
    ├── gradio_qwen_refined.py  # Earlier implementation
    └── gradio_qwen_debug.py    # Debug version
```

### Key Classes

- **`Qwen3ConfigurationManager`**: Device and profile-specific configurations
- **`EnhancedQwen3Streamer`**: Real-time response streaming with token filtering
- **`SystemMetrics`**: Comprehensive performance monitoring
- **`Qwen3NPUDeployment`**: Specialized NPU deployment with error handling

## 🔧 Advanced Configuration

### NPU Optimization

The application includes comprehensive NPUW configuration for Intel NPU:

```python
npu_config = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES", 
    "NPUW_LLM_MAX_PROMPT_LEN": 2048,
    "NPUW_LLM_MIN_RESPONSE_LEN": 256,
    "NPUW_LLM_PREFILL_HINT": "BEST_PERF",
    "NPUW_LLM_GENERATE_HINT": "BEST_PERF"
}
```

### Custom Profiles

Create custom optimization profiles:

```python
custom_profile = NPUPerformanceProfile(
    max_prompt_len=4096,
    min_response_len=512, 
    cache_mode="OPTIMIZE_SPEED",
    performance_hint="LATENCY",
    compilation_strategy="OPTIMAL"
)
```

## 📊 Performance Monitoring

Access real-time performance metrics through the UI:
- **Response times** and token generation rates
- **Device utilization** and memory usage  
- **Error rates** and compilation diagnostics
- **Token filtering** statistics

## 🛠️ Troubleshooting

### Common Issues

**NPU Compilation Errors:**
```bash
# Check NPUW configuration
export NPUW_LLM=YES
export NPU_USE_NPUW=YES

# Verify driver installation
ov-device-list
```

**Memory Issues:**
- Reduce `MAX_PROMPT_LEN` in configuration
- Use "conservative" NPU profile
- Enable `NPU_LOW_MEMORY_MODE`

**Slow Performance:**
- Verify NPU drivers are installed
- Check cache directory permissions
- Use "aggressive" profile for maximum speed

### Debug Mode

Enable detailed logging:
```python
# In gradio_qwen_enhanced.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions are welcome! This is a hobby project aimed at showcasing OpenVINO GenAI capabilities.

### Areas for Enhancement
- Additional model support (Llama, ChatGLM, etc.)
- Multi-modal capabilities (vision, audio)
- Distributed inference support
- Advanced UI features

### Development Setup
1. Fork the repository
2. Create feature branch
3. Follow existing code patterns
4. Add tests for new functionality
5. Submit pull request

## 📚 Documentation

- **[ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md)**: Third-party attributions
- **[DISCLAIMER.md](DISCLAIMER.md)**: Usage guidelines and disclaimers
- **[CLAUDE.md](CLAUDE.md)**: Development guidelines and context
- **`context/README.md`**: Enhanced context system documentation

## 🔍 Examples

### Simple Chat
```python
# Basic usage
message = "Explain quantum computing"
response = enhanced_qwen3_chat(message, [])
```

### Performance Monitoring
```python
# Access metrics
metrics = system_metrics.to_display_dict()
print(f"Avg response time: {metrics['Avg Response Time']}")
```

### Custom Configuration  
```python
# Deploy with custom config
deployment = Qwen3NPUDeployment(model_path, "aggressive")
pipeline = deployment.deploy()
```

## 📈 Performance Targets

### Intel NPU (Typical)
- **Load Time**: 60-90 seconds (first run), 30-60 seconds (cached)
- **First Token**: <2 seconds latency
- **Generation**: 15-25 tokens/second
- **Memory**: Optimized for NPU constraints

### CPU Fallback
- **Load Time**: 10-30 seconds
- **Generation**: 8-15 tokens/second  
- **Memory**: 8GB+ recommended

## 🔄 Version History

- **v1.0 (Enhanced)**: Production-ready with complete optimization
- **v0.9 (Refined)**: Hybrid architecture with consultant insights
- **v0.8 (Debug)**: NPU-optimized with debugging features
- **v0.7**: Basic Gradio integration

## ⚖️ Legal & Licensing

### License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Components
This project integrates with several third-party components, each with their own licenses:
- **OpenVINO GenAI**: Apache 2.0 License (Intel Corporation)
- **Qwen3 Models**: Apache 2.0 with additional terms (Alibaba Cloud)  
- **Gradio**: Apache 2.0 License (HuggingFace)
- **Transformers**: Apache 2.0 License (HuggingFace)

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for detailed attributions.

### Model Usage
This software is designed to work with Qwen3-8B models. Users must:
1. **Obtain Qwen3 models** through official channels (HuggingFace, ModelScope, etc.)
2. **Comply with Qwen model license terms** including attribution requirements
3. **Follow responsible AI usage guidelines**

### Disclaimer
This is a **hobby/educational project** and is not intended for commercial use. Users are responsible for ensuring compliance with all applicable licenses and terms of service.

See [DISCLAIMER.md](DISCLAIMER.md) for complete usage guidelines.

## 🙏 Acknowledgments

This project builds upon the excellent work of many open-source contributors:

- **Intel Corporation** for OpenVINO GenAI and NPU optimization techniques
- **Alibaba Cloud/Qwen Team** for the Qwen3 model architecture and specifications
- **HuggingFace** for Gradio, Transformers, and the broader AI ecosystem
- **The open-source AI community** for sharing knowledge and best practices

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for complete attributions.

---

## 🔗 Links

- **OpenVINO GenAI**: https://github.com/openvinotoolkit/openvino.genai
- **Qwen Models**: https://github.com/QwenLM/Qwen  
- **Gradio**: https://gradio.app/
- **Intel NPU**: https://www.intel.com/content/www/us/en/products/docs/processors/core/core-processors-with-intel-npu.html

---

*⭐ If this project helps you, please consider giving it a star!*

*🐛 Found a bug? Please open an issue!*

*💡 Have an idea? Contributions are welcome!*