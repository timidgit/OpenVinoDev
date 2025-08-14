# Enhanced Phi-3 OpenVINO GenAI Chat Application

A production-ready, modular implementation of microsoft/Phi-3-mini-128k-instruct chat interface using OpenVINO GenAI with Intel NPU optimization, RAG document processing, and comprehensive performance monitoring. Built with professional software engineering practices and a modular architecture for maximum maintainability and scalability.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![OpenVINO](https://img.shields.io/badge/OpenVINO-2024.4+-orange.svg)

## ✨ Features

### 🚀 **Performance Optimization**
- **Complete Intel NPU optimization** with NPUW (NPU Wrapper) configuration
- **Phi-3-specific tuning** with model architecture awareness and 128k context support
- **Intelligent device fallback** (NPU → CPU) with appropriate configurations
- **Professional performance monitoring** with real-time metrics
- **Fixed NPUW configuration issues** for stable NPU compilation

### 🎯 **Advanced Capabilities**
- **🎯 Dynamic System Prompts**: Real-time AI behavior customization
- **📚 RAG Document Processing**: Upload and query your own documents
- **🔍 Special Token Filtering** for clean Phi-3 output
- **📊 Real-time Performance Monitoring** with comprehensive metrics
- **⚙️ Multi-source Configuration** (CLI, env vars, JSON)

### 🖥️ **Professional User Experience**
- **Modern Gradio interface** with professional theming and accordions
- **Real-time streaming responses** with token-level filtering
- **Smart message processing** with intelligent truncation
- **Document upload and processing** for context-aware conversations
- **Security-focused input validation** and sanitization

### 🏗️ **Enterprise-Ready Architecture**
- **Modular codebase** with clear separation of concerns
- **Command-line interface** with comprehensive argument support
- **Configuration management** with environment override support
- **Professional error handling** with detailed diagnostics
- **Comprehensive logging** and debugging capabilities

## 🛠️ Installation

### Prerequisites
- **Python 3.8+**
- **OpenVINO 2024.4+** with GenAI support
- **Intel NPU drivers** (for NPU acceleration)
- **microsoft/Phi-3-mini-128k-instruct model** in OpenVINO format

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
export MODEL_PATH="/path/to/your/phi3-128k-npu"
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

**Option C: Command-Line Arguments**
```bash
# Use CLI arguments for quick configuration changes
python main.py --device CPU --npu-profile conservative
```

5. **Run the application**
```bash
# Modular entry point (unified architecture)
python main.py
```

## 📋 Requirements

### Hardware
- **Recommended**: Intel NPU-enabled system (Meteor Lake, Arrow Lake, or later)
- **Alternative**: Intel CPU with AVX-512 support
- **Memory**: 8GB+ RAM for optimal performance

### Software
**Core Dependencies:**
```
openvino-genai>=2024.4
gradio>=4.0.0
transformers>=4.30.0
numpy>=1.21.0
typing-extensions>=4.0.0
```

**RAG Dependencies (Optional):**
```
langchain>=0.1.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
```

See `requirements.txt` for complete dependencies.

## 🚀 Usage

### Basic Usage
```bash
# Start with default configuration
python main.py

# Use specific device
python main.py --device CPU

# Custom model path and profile
python main.py --model-path ./models/phi3-128k-npu --npu-profile balanced

# Enable public sharing (use with caution)
python main.py --share

# Show all available options
python main.py --help
```

The application will:
1. **Auto-detect** available hardware (NPU/CPU)
2. **Deploy** Phi-3 model with optimal configuration
3. **Launch** web interface at `http://127.0.0.1:7860`

### Configuration Options

**Environment Variables:**
```bash
export MODEL_PATH="/path/to/model"            # Model location
export CACHE_DIR="/path/to/cache"             # Cache directory
export NPU_PROFILE="balanced"                 # balanced|conservative|aggressive
export MAX_MESSAGE_LENGTH="2000"              # Max input length (increased for Phi-3 128k context)
```

**NPU Profiles:**
- **Conservative**: Lower memory usage, smaller contexts
- **Balanced**: Optimal for most use cases (default)
- **Aggressive**: Maximum performance, higher memory usage

## 🏗️ Architecture

### Core Components

```
OpenVinoDev/
├── app/                        # Modular application architecture
│   ├── __init__.py            # Package initialization & exports
│   ├── config.py              # Configuration management
│   ├── model.py               # Model deployment & system initialization
│   ├── streamer.py            # Token streaming & filtering
│   ├── chat.py                # Core chat processing & RAG
│   └── ui.py                  # Gradio interface & event handling
├── _context_archive/          # Archived legacy code and context files
│   ├── archive/               # Legacy implementations
│   ├── context/               # Reference examples and patterns
│   ├── tests/                 # Unit tests
│   └── documentation/         # Additional docs and licenses
├── main.py                    # Application entry point with CLI
├── config.json               # Configuration file
└── requirements.txt          # Dependencies
```

### Key Components

**Configuration Management (`app/config.py`)**
- **`ConfigurationLoader`**: Multi-source configuration (JSON, env vars, CLI)
- **JSON/Environment/CLI priority system** for flexible configuration

**Model Management (`app/model.py`)**
- **`deploy_qwen3_pipeline()`**: NPU-optimized model deployment (legacy naming)
- **`initialize_system_with_validation()`**: Comprehensive system initialization
- **Configuration management**: Device and profile-specific configurations for Phi-3

**Streaming & Performance (`app/streamer.py`)**
- **`EnhancedQwen3Streamer`**: Real-time response streaming with token filtering (legacy naming)
- **`StreamingMetrics`**: Performance monitoring and diagnostics

**Chat Processing (`app/chat.py`)**
- **`enhanced_qwen3_chat()`**: Main chat processing with RAG integration (legacy naming)
- **`DocumentRAGSystem`**: Document upload and context retrieval
- **`InputValidator`**: Security-focused input validation

**User Interface (`app/ui.py`)**
- **`create_enhanced_interface()`**: Professional Gradio interface
- **Dynamic system prompts, RAG uploads, performance monitoring**

## 🔧 Advanced Configuration

### NPU Optimization

The application includes comprehensive NPUW configuration for Intel NPU:

```python
npu_config = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES", 
    "NPUW_LLM_MAX_PROMPT_LEN": 8192,  # Increased for Phi-3 128k context
    "NPUW_LLM_MIN_RESPONSE_LEN": 256,
    "NPUW_LLM_PREFILL_HINT": "LATENCY",
    "NPUW_LLM_GENERATE_HINT": "LATENCY"
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
```bash
# Use debug flag with modular version
python main.py --debug

```

### Command-Line Diagnostics

The new modular version includes comprehensive CLI diagnostics:

```bash
# Validate system requirements only
python main.py --validate-only

# Debug mode with verbose logging
python main.py --debug

# Test specific configurations
python main.py --device CPU --debug
python main.py --npu-profile conservative --debug
```

## 🎯 New Features Guide

### Dynamic System Prompts

Customize AI behavior in real-time:

1. **Access**: Click "🎯 System Prompt Configuration" in the interface
2. **Edit**: Modify the prompt to change AI behavior, expertise, and style
3. **Apply**: Click "✅ Apply & Clear Chat" to activate changes
4. **Reset**: Use "🔄 Reset to Default" to restore original prompt

**Example Custom Prompt:**
```
You are a Python coding expert specializing in data science and machine learning.

Key behaviors:
- Provide complete, working code examples
- Explain complex algorithms step-by-step
- Include relevant imports and dependencies
- Focus on best practices and optimization

You excel at: pandas, numpy, scikit-learn, and deep learning frameworks.
```

### RAG Document Processing

Upload and query your own documents:

1. **Upload**: Click "📚 Document Upload (RAG)" and select files
2. **Supported**: `.txt`, `.md`, `.py`, `.js`, `.html`, `.css`, `.json` files
3. **Process**: Wait for "✅ Successfully processed" confirmation
4. **Query**: Ask questions about your uploaded content
5. **Context**: AI will reference relevant document sections in responses

**Example Queries:**
- "What does the uploaded document say about error handling?"
- "Summarize the key points from the technical specification"
- "How should I implement the feature described in the documentation?"

### Performance Monitoring

Access real-time metrics:

1. **View**: Click "📊 Performance Metrics" 
2. **Monitor**: Real-time response times, tokens/second, device utilization
3. **Analyze**: Special tokens filtered, compilation errors, cache hits
4. **Reset**: Use "🔄 Reset Metrics" to clear historical data

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
response = enhanced_qwen3_chat(message, [])  # Legacy function naming
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
from app.npu_patterns import initialize_phi3_pipeline
pipeline = initialize_phi3_pipeline(model_path, profile="aggressive")
```

## 📈 Performance Targets

### Intel NPU (Typical)
- **Load Time**: 60-90 seconds (first run), 30-60 seconds (cached)
- **First Token**: <2 seconds latency
- **Generation**: 10-20 tokens/second (Phi-3 is smaller than Qwen3, slightly different performance)
- **Memory**: Optimized for NPU constraints
- **Context**: Up to 128k tokens supported

### CPU Fallback
- **Load Time**: 10-25 seconds (Phi-3 is 3.8B vs 8B parameters)
- **Generation**: 5-12 tokens/second  
- **Memory**: 6GB+ recommended (lower than Qwen3 due to smaller model size)

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
- **Phi-3 Models**: MIT License (Microsoft Corporation)
- **Gradio**: Apache 2.0 License (HuggingFace)
- **Transformers**: Apache 2.0 License (HuggingFace)

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for detailed attributions.

### Model Usage
This software is designed to work with microsoft/Phi-3-mini-128k-instruct. Users must:
1. **Obtain Phi-3 models** through official channels (HuggingFace, etc.)
2. **Comply with Phi-3 model license terms** (MIT License with responsible AI guidelines)
3. **Follow responsible AI usage guidelines**

### Disclaimer
This is a **hobby/educational project** and is not intended for commercial use. Users are responsible for ensuring compliance with all applicable licenses and terms of service.

See [DISCLAIMER.md](DISCLAIMER.md) for complete usage guidelines.

## 🙏 Acknowledgments

This project builds upon the excellent work of many open-source contributors:

- **Intel Corporation** for OpenVINO GenAI and NPU optimization techniques
- **Microsoft Corporation** for the Phi-3 model architecture and specifications
- **HuggingFace** for Gradio, Transformers, and the broader AI ecosystem
- **The open-source AI community** for sharing knowledge and best practices

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for complete attributions.

---

## 🔗 Links

- **OpenVINO GenAI**: https://github.com/openvinotoolkit/openvino.genai
- **Phi-3 Models**: https://huggingface.co/microsoft/Phi-3-mini-128k-instruct  
- **Gradio**: https://gradio.app/
- **Intel NPU**: https://www.intel.com/content/www/us/en/products/docs/processors/core/core-processors-with-intel-npu.html

---

*⭐ If this project helps you, please consider giving it a star!*

*🐛 Found a bug? Please open an issue!*

*💡 Have an idea? Contributions are welcome!*