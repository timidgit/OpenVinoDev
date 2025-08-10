# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## **Project Overview**

This is a **production-ready, modular Qwen3-8B chat application** using OpenVINO GenAI with Intel NPU optimization. The project has evolved from a monolithic implementation to a professional modular architecture with comprehensive features including RAG document processing, dynamic system prompts, and advanced performance monitoring.

## **Architecture Overview**

### **Dual Entry Points**
- **`main.py`** (RECOMMENDED): Modern modular architecture with CLI interface
- **`gradio_qwen_enhanced.py`**: Legacy single-file version (maintained for compatibility)

**CRITICAL**: All new development should target the **modular architecture** (`app/` directory) accessed via `main.py`.

### **Core Module Structure**
```
app/
├── config.py      # Multi-source configuration (CLI → env → JSON → defaults)
├── model.py       # NPU pipeline deployment with comprehensive fallbacks  
├── streamer.py    # Qwen3-specific token filtering & performance metrics
├── chat.py        # Chat processing with RAG integration & Gradio compatibility
└── ui.py          # Professional Gradio interface with advanced features
```

### **Context System**
```
context/
├── qwen3_model_context/     # Qwen3-specific optimizations
│   ├── npu_optimization.py  # NPU NPUW configuration profiles
│   ├── model_architecture.py # Model-specific settings
│   └── special_tokens.py    # Token filtering definitions
├── core_cpp/               # OpenVINO GenAI C++ insights
├── python_samples/         # Reference implementations
└── gradio_patterns/        # UI component patterns
```

## **Common Commands**

### **Primary Application Launch**
```bash
# Modern modular entry point (PREFERRED)
python main.py

# Legacy single-file version (compatibility)
python gradio_qwen_enhanced.py
```

### **CLI Configuration & Testing**
```bash
# System validation
python main.py --validate-only

# Device-specific testing
python main.py --device CPU
python main.py --device NPU --npu-profile conservative

# Debug mode with verbose logging  
python main.py --debug

# Custom configuration
python main.py --model-path ./models/qwen3 --port 8080

# Full CLI help
python main.py --help
```

### **Development & Utilities**
```bash
# Export NPU-compatible model
python export_qwen_for_npu.py --model Qwen/Qwen2-7B-Instruct --output qwen2-7b-npu

# Analyze model compatibility  
python check_model_config.py

# Create consolidated context
create_llm_context.bat
```

## **Critical Configuration Knowledge**

### **NPUW Configuration (ESSENTIAL)**
The NPU requires specific NPUW hint values. **Recent fixes corrected unsupported values**:

```python
# CORRECT (Fixed in codebase):
"NPUW_LLM_PREFILL_HINT": "FAST_COMPILE",
"NPUW_LLM_GENERATE_HINT": "FAST_COMPILE"

# INCORRECT (Causes compilation errors):
"NPUW_LLM_PREFILL_HINT": "BEST_PERF",  # Not supported by current drivers
"NPUW_LLM_GENERATE_HINT": "BEST_PERF"  # Not supported by current drivers
```

### **Configuration Priority System**
The modular architecture implements **4-tier configuration priority**:
1. **CLI Arguments** (highest): `--device CPU --npu-profile conservative`
2. **Environment Variables**: `QWEN3_MODEL_PATH=/path/to/model`
3. **JSON Configuration**: `config.json` settings
4. **Built-in Defaults** (lowest): Hardcoded fallbacks

### **NPU Deployment Strategy**
`deploy_qwen3_pipeline()` implements multi-tier fallback:
1. **Enhanced Context**: Uses `context/qwen3_model_context/` optimizations
2. **Manual Configuration**: Fallback NPUW settings
3. **Basic Configuration**: Minimal OpenVINO properties  
4. **CPU Fallback**: Automatic device switching

## **Gradio ChatInterface Compatibility**

### **Data Format Requirements (CRITICAL)**
The chat system uses **Gradio ChatInterface** which expects specific data formats:

```python
# CORRECT format for Gradio ChatInterface:
ChatHistory = List[List[str]]  # [["user_msg", "bot_response"], ...]

# Functions return/yield this format:
history = [["Hello", "Hi there!"], ["How are you?", "I'm doing well!"]]

# WRONG format (causes Gradio errors):
history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]
```

### **Streaming Implementation**
```python
# Correct streaming pattern for Gradio:
def stream_response_to_history(streamer, history):
    for chunk in streamer:
        if chunk:
            history[-1][1] += chunk  # Update bot response part
            yield history
```

## **Stateful API Usage (ESSENTIAL)**

OpenVINO GenAI pipelines are **stateful** - they manage conversation history internally:

```python
# CORRECT Pattern:
pipe.start_chat(SYSTEM_PROMPT)        # Initialize session
pipe.generate(new_message, config)    # Send only new message  
pipe.finish_chat()                    # Clear session

# WRONG Pattern (causes token limit errors):
full_conversation = build_history(history + [new_message])
prompt = tokenizer.apply_chat_template(full_conversation)
pipe.generate(prompt)  # Re-processes entire conversation each time
```

## **Advanced Features Integration**

### **RAG Document Processing**
- **Dependencies**: `langchain`, `faiss-cpu`, `sentence-transformers`
- **Supported formats**: `.txt`, `.md`, `.py`, `.js`, `.html`, `.css`, `.json`
- **Security**: File type validation, size limits, content sanitization
- **Integration**: Automatic context retrieval and prompt augmentation

### **Dynamic System Prompts**
- **Real-time customization** through Gradio interface
- **Session management**: `pipe.start_chat(SYSTEM_PROMPT)`
- **Persistence**: Maintained across conversation until explicitly changed

### **Performance Monitoring**
- **Real-time metrics**: Token generation rates, response times, device utilization
- **NPU-specific diagnostics**: Compilation status, memory usage, fallback triggers
- **Streaming metrics**: Token filtering statistics, special token handling

## **Development Guidelines**

### **NPU Development Constraints**
- **Token Limits**: NPU has hard-coded prompt length limits (~2048 tokens)
- **Defensive Programming**: Always validate input length before NPU processing
- **Configuration Validation**: Ensure NPUW settings are supported
- **3-Strike Rule**: After 3 NPU failures, reassess approach and use CPU fallback

### **Security-First Development**
- **Input Validation**: All user inputs processed through `InputValidator` class
- **File Upload Security**: Type restrictions, size limits, content validation
- **Environment Configuration**: Sensitive data via environment variables only
- **Error Handling**: User-friendly messages without exposing internals

### **Modular Development Process**
1. **Target `app/` modules** for all new features
2. **Use `main.py` CLI interface** for testing and configuration
3. **Follow configuration priority system** (CLI → env → JSON → defaults)
4. **Implement comprehensive error handling** with device fallbacks
5. **Maintain Gradio data format compatibility** in chat functions

### **Testing Strategy**
```bash
# Validate system requirements
python main.py --validate-only

# Test NPU functionality with debug output
python main.py --device NPU --debug

# Test CPU fallback behavior
python main.py --device CPU --debug

# Test configuration loading
python main.py --config custom_config.json --debug
```

## **Legacy Files (Archive Only)**

**DO NOT USE PATTERNS FROM THESE FILES**:
- `archive/gradio_qwen_*.py` - Historical implementations
- These are maintained for reference only, not for new development
- All functionality has been migrated to the modular architecture

## **Quality Gates for NPU Features**

A feature is "production ready" when:
- ✅ NPU compilation succeeds with target configuration
- ✅ CPU fallback works when NPU fails
- ✅ User receives clear feedback about active device
- ✅ Performance metrics are reasonable (>10 tokens/sec for NPU)
- ✅ Memory usage stays within NPU constraints
- ✅ Long conversations don't cause token limit crashes
- ✅ Gradio data format compatibility maintained

## **Critical Implementation Patterns**

### **Configuration Composition**
- Use dependency injection, not global variables
- `ConfigurationLoader` manages all settings
- Environment-agnostic code for deployment flexibility
- Every feature has CPU/basic fallback option

### **Error Handling Strategy**
- Fail fast with descriptive messages
- Specific try/except blocks with clear fallback paths
- User-friendly error messages without technical details
- Device fallback triggers with transparent communication

### **Performance Optimization**
- Token-level streaming with Qwen3-specific filtering
- Real-time metrics integration throughout pipeline
- Model compilation caching for faster subsequent loads
- Proper resource cleanup and session management