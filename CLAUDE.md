# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## **Project Overview**

This is a **production-ready, modular Phi-3-mini-128k-instruct chat application** using OpenVINO GenAI with Intel NPU optimization. The project has evolved from a monolithic implementation to a professional modular architecture with comprehensive features including RAG document processing, dynamic system prompts, and advanced performance monitoring. **Recently migrated from Qwen3-8B to Phi-3 for massive 128k context support**.

## **Architecture Overview**

### **Unified Architecture**
- **`main.py`**: Modular architecture with CLI interface (unified entry point)
- **`archive/gradio_qwen_enhanced.py`**: Archived legacy implementation

**CRITICAL**: All development targets the **modular architecture** (`app/` directory) accessed via `main.py`.

### **Core Module Structure**
```
app/
├── config.py      # Multi-source configuration (CLI → env → JSON → defaults)
├── model.py       # NPU pipeline deployment with comprehensive fallbacks  
├── streamer.py    # Token filtering & performance metrics (legacy Qwen3 naming)
├── chat.py        # Chat processing with RAG integration & Gradio compatibility
└── ui.py          # Professional Gradio interface with advanced features
```

### **Context System**
```
context/
├── qwen3_model_context/     # Legacy optimizations (still used for NPU patterns)
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
# Modular entry point (unified architecture)
python main.py
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

# Custom model path (useful for model switching)
python main.py --model-path "C:\OpenVinoModels\phi3-128k-npu"

# Full CLI help
python main.py --help
```

### **Development & Utilities**
```bash
# Export NPU-compatible model
python export_qwen_for_npu.py --model microsoft/Phi-3-mini-128k-instruct --output phi3-128k-npu

# Analyze model compatibility  
python check_model_config.py

# Create consolidated context
create_llm_context.bat
```

### **Installation**
```bash
# Install dependencies
pip install -r requirements.txt

# Key dependencies
pip install openvino-genai>=2024.4.0 gradio>=4.0.0 transformers>=4.30.0

# RAG dependencies (optional)
pip install langchain faiss-cpu sentence-transformers
```

## **Current Model Configuration**

### **Phi-3-mini-128k-instruct (Current)**
The application now uses **microsoft/Phi-3-mini-128k-instruct** with massive context improvements:

```json
{
  "model": {
    "path": "C:\\OpenVinoModels\\phi3-128k-npu",
    "name": "Phi-3-mini-128k-instruct", 
    "type": "phi3"
  },
  "ui": {
    "max_message_length": 2000,      # Was 400 (5x increase)
    "max_conversation_tokens": 8000, # Was 1800 (4.4x increase)
    "emergency_limit": 16384         # Was 2048 (8x increase)
  }
}
```

### **Environment Variable Configuration**
```bash
# Primary (recommended)
set MODEL_PATH=C:\OpenVinoModels\phi3-128k-npu

# Backward compatibility  
set QWEN3_MODEL_PATH=C:\OpenVinoModels\phi3-128k-npu

# Device and profile
set TARGET_DEVICE=NPU
set NPU_PROFILE=balanced
```

## **Critical NPU Configuration Knowledge**

### **NPUW Configuration (ESSENTIAL)**
NPU requires specific NPUW hint values. **Current configuration uses supported values**:

```python
# CORRECT (Currently in codebase after fixes):
"NPUW_LLM_PREFILL_HINT": "LATENCY",
"NPUW_LLM_GENERATE_HINT": "LATENCY",
"NPUW_LLM_MAX_PROMPT_LEN": 8192,  # Increased for Phi-3 128k context

# INCORRECT (Causes compilation errors):
"NPUW_LLM_PREFILL_HINT": "FAST_COMPILE",  # Not supported by current drivers
"NPUW_LLM_GENERATE_HINT": "BEST_PERF"     # Not supported

# CRITICAL: Do NOT use generic PERFORMANCE_HINT with NPUW hints - they conflict
```

### **Configuration Priority System**
The modular architecture implements **4-tier configuration priority**:
1. **CLI Arguments** (highest): `--device CPU --model-path path/to/model`
2. **Environment Variables**: `MODEL_PATH=/path/to/model`
3. **JSON Configuration**: `config.json` settings
4. **Built-in Defaults** (lowest): Hardcoded fallbacks

### **NPU Deployment Strategy**
`deploy_qwen3_pipeline()` (function name retained for compatibility) implements multi-tier fallback:
1. **Enhanced Context**: Uses `context/qwen3_model_context/` for NPU optimization patterns
2. **Manual Configuration**: Fallback NPUW settings optimized for Phi-3
3. **Basic Configuration**: Minimal OpenVINO properties  
4. **CPU Fallback**: Automatic device switching

## **Gradio ChatInterface Compatibility (CRITICAL)**

### **Data Format Requirements**
The chat system uses **Gradio ChatInterface** which requires specific data formats:

```python
# CORRECT format for Gradio ChatInterface:
ChatHistory = List[List[str]]  # [["user_msg", "bot_response"], ...]

# Functions return/yield this format:
history = [["Hello", "Hi there!"], ["How are you?", "I'm doing well!"]]

# WRONG format (causes "Data incompatible with messages format" errors):
history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]
```

### **Streaming Implementation**
```python
# Correct streaming pattern for Gradio:
def stream_response_to_history(streamer, history):
    for chunk in streamer:
        if chunk:
            history[-1][1] += chunk  # Update bot response part (index [1])
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

### **Model Migration Patterns**
When switching models:
1. **Update config.json**: Model path, cache directory, token limits
2. **Update app/model.py**: NPU configuration limits, cache paths, print statements
3. **Update app/ui.py**: UI labels, system prompts, performance targets
4. **Update main.py**: Application descriptions and startup messages
5. **Test thoroughly**: Validate system requirements and full startup

### **NPU Development Constraints**
- **Token Limits**: NPU has hard-coded prompt length limits (now 8192 for Phi-3)
- **Defensive Programming**: Always validate input length before NPU processing
- **Configuration Validation**: Ensure NPUW settings use supported values
- **Conflict Avoidance**: Do not use generic PERFORMANCE_HINT with NPUW-specific hints

### **Testing Strategy**
```bash
# Validate system requirements
python main.py --validate-only

# Test NPU functionality with debug output
python main.py --device NPU --debug

# Test CPU fallback behavior  
python main.py --device CPU --debug

# Test model path override
python main.py --model-path "C:\OpenVinoModels\phi3-128k-npu" --debug
```

## **Legacy Context System**

**NOTE**: The `context/qwen3_model_context/` directory retains its Qwen3 naming but contains NPU optimization patterns that work effectively with Phi-3. The enhanced context system:
- Provides NPU NPUW configuration profiles with correct hint values
- Contains C++ reference implementations
- Includes Gradio integration patterns
- Should be preserved even when migrating to other models

## **Quality Gates for NPU Features**

A feature is "production ready" when:
- ✅ NPU compilation succeeds with target configuration
- ✅ CPU fallback works when NPU fails
- ✅ User receives clear feedback about active device
- ✅ Performance metrics are reasonable (>10 tokens/sec for NPU)
- ✅ Memory usage stays within NPU constraints
- ✅ Conversations leverage full context without crashes
- ✅ Gradio data format compatibility maintained

## **Critical Implementation Patterns**

### **Configuration Composition**
- Use dependency injection, not global variables
- `ConfigurationLoader` manages all settings with priority system
- Environment-agnostic code for deployment flexibility
- Every feature has CPU/basic fallback option

### **Error Handling Strategy**
- Fail fast with descriptive messages
- Specific try/except blocks with clear fallback paths
- User-friendly error messages without technical details
- Device fallback triggers with transparent communication

### **Performance Optimization**
- Token-level streaming with model-agnostic filtering
- Real-time metrics integration throughout pipeline
- Model compilation caching for faster subsequent loads
- Proper resource cleanup and session management
- Leverage Phi-3's 128k context for complex conversations