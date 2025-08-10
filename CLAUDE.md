# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## **Architecture Overview**

This project has **evolved from a monolithic implementation to a production-ready modular architecture**. The codebase represents a complete transformation following professional software engineering practices.

### **Current State: Dual Architecture**

The repository supports **two entry points**:

1. **`main.py` (RECOMMENDED)**: Modern modular architecture with CLI interface
2. **`gradio_qwen_enhanced.py`**: Legacy single-file version (still functional)

**CRITICAL**: All new development should target the **modular architecture** (`app/` directory) accessed via `main.py`.

### **Modular Application Structure**

```
app/
├── config.py      # Multi-source configuration (CLI → env → JSON → defaults)
├── model.py       # NPU pipeline deployment with comprehensive fallbacks  
├── streamer.py    # Qwen3-specific token filtering & performance metrics
├── chat.py        # Chat processing with RAG integration
└── ui.py          # Professional Gradio interface with advanced features
```

### **Legacy Files (Archive Only)**

**DO NOT USE PATTERNS FROM THESE FILES**:
- `archive/gradio_qwen_*.py` - Historical implementations, moved to archive
- Use only as reference for understanding evolution, not for new code

## **Common Commands**

### **Primary Application Launch**
```bash
# Modern modular entry point (PREFERRED)
python main.py

# Legacy single-file version
python gradio_qwen_enhanced.py
```

### **Configuration & Development**
```bash
# System validation
python main.py --validate-only

# Debug mode with verbose logging  
python main.py --debug

# Device-specific testing
python main.py --device CPU
python main.py --device NPU --npu-profile conservative

# Custom model path
python main.py --model-path ./models/qwen3

# Full CLI help
python main.py --help
```

### **Context and Utilities**
```bash
# Export NPU-compatible model
python export_qwen_for_npu.py --model Qwen/Qwen2-7B-Instruct --output qwen2-7b-npu

# Analyze model compatibility  
python check_model_config.py

# Create consolidated context
create_llm_context.bat
```

## **NPU Configuration (CRITICAL)**

### **Fixed Configuration Issues**

The NPU NPUW hints have been **corrected from problematic values**:

```python
# CORRECT (Fixed in codebase):
"NPUW_LLM_PREFILL_HINT": "LATENCY",
"NPUW_LLM_GENERATE_HINT": "LATENCY"

# INCORRECT (Would cause compilation errors):
"NPUW_LLM_PREFILL_HINT": "BEST_PERF",  # Not supported by current drivers
"NPUW_LLM_GENERATE_HINT": "BEST_PERF"  # Not supported by current drivers
```

### **Configuration Priority System**

The modular architecture implements a **4-tier configuration priority**:

1. **CLI Arguments** (highest priority): `--device CPU --npu-profile conservative`
2. **Environment Variables**: `TARGET_DEVICE=NPU NPU_PROFILE=balanced`
3. **JSON Configuration**: `config.json` file settings
4. **Built-in Defaults** (lowest priority): Hardcoded fallbacks

### **NPU Deployment Strategy**

The `deploy_qwen3_pipeline()` function implements a **multi-tier fallback**:

1. **Enhanced Context**: Uses `context/qwen3_model_context/` optimizations
2. **Manual Configuration**: Fallback NPUW settings
3. **Basic Configuration**: Minimal OpenVINO properties
4. **CPU Fallback**: Automatic device switching

## **Advanced Features Integration**

### **RAG Document Processing**

The modular system includes **production-ready RAG integration**:

- **Document Upload**: `.txt`, `.md`, `.py`, `.js`, `.html`, `.css`, `.json` support
- **Vector Storage**: FAISS-based similarity search
- **Context Retrieval**: Automatic relevant document section extraction
- **Security**: Input validation and file type restrictions

### **Dynamic System Prompts** 

**Real-time AI behavior customization**:
- User-editable system prompts through Gradio interface
- Session management with `pipe.start_chat(SYSTEM_PROMPT)`
- Reset to defaults and apply changes instantly

### **Performance Monitoring**

**Comprehensive metrics tracking**:
- Real-time response times and token generation rates
- Device utilization and NPU-specific diagnostics  
- Special token filtering statistics
- Professional JSON-based metrics display

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

## **Development Guidelines**

### **Modular Development Process**

1. **Target the `app/` modules** for all new features
2. **Use `main.py` CLI interface** for testing and configuration
3. **Follow the configuration priority system** (CLI → env → JSON → defaults)  
4. **Implement comprehensive error handling** with device fallbacks
5. **Add security validation** for user inputs and file uploads

### **NPU Development Constraints**

- **Token Limits**: NPU has hard-coded prompt length limits (~2048 tokens)
- **Defensive Programming**: Always validate input length before NPU processing
- **Compilation Errors**: NPU compilation failures require specific NPUW configuration
- **3-Strike Rule**: After 3 NPU failures, reassess approach and use CPU fallback

### **Testing Strategy**

```bash
# Test NPU functionality
python main.py --device NPU --debug

# Test CPU fallback  
python main.py --device CPU --debug

# Validate system requirements
python main.py --validate-only

# Test with custom configuration
python main.py --config custom_config.json --debug
```

## **Critical Implementation Patterns**

### **Security-First Development**

- **Input Validation**: All user inputs go through `InputValidator` class
- **File Upload Security**: Restricted file types, size limits, content validation
- **Environment Variables**: All paths and sensitive data use env vars
- **Error Handling**: Specific exceptions with user-friendly messages

### **Configuration Composition**

- **No Global Variables**: Use dependency injection pattern
- **Configuration Objects**: `ConfigurationLoader` manages all settings  
- **Environment Agnostic**: Code works across different deployment scenarios
- **Fallback Safety**: Every feature has CPU/basic fallback option

### **Performance Optimization**

- **Streaming Architecture**: Token-level response streaming with filtering
- **Metrics Integration**: Real-time performance monitoring throughout
- **Caching Strategy**: Model compilation cache for faster subsequent loads
- **Resource Management**: Proper cleanup and session management

## **2\. Project Architecture: An Evolutionary Path**

This project documents the evolution of an OpenVINO GenAI chat application, culminating in a **single, production-grade implementation**. The architecture is not a set of parallel options, but a progression.

### **The Canonical Implementation: gradio\_qwen\_enhanced.py**

This is the **definitive version**. Its architecture correctly integrates all critical systems and should be the sole inspiration for any new code.

* **Primary Strengths**:  
  * **Enhanced Context Integration**: Correctly imports and utilizes specialized modules from the context/ directory.  
  * **Graceful Fallback Mechanism**: Uses the ENHANCED\_CONTEXT\_AVAILABLE pattern, allowing the application to run with limited features if the full context is missing.  
  * **Qwen-Specific Optimization**: Demonstrates deep model-specific tuning, including professional-grade filtering for over 26 special tokens (QwenStreamingFilter).  
  * **Correct Stateful API Usage**: Properly uses pipe.start\_chat() and pipe.finish\_chat() for robust session management, as shown in python\_samples/chat\_sample.py.  
  * **Comprehensive Diagnostics**: Implements a SystemMetrics dataclass and a professional Gradio UI with real-time performance monitoring.  
  * **Robust Deployment**: The deploy\_qwen\_pipeline function shows a multi-tiered fallback strategy (enhanced → basic → minimal → CPU).

### **Legacy Implementations (For Historical Context ONLY)**

**These files are read-only artifacts. Do not use their patterns.**

* **gradio\_qwen\_refined.py**: The first version to correctly implement the stateful API. Its logic has been **superseded and improved upon** in the enhanced version.  
* **gradio\_qwen\_hybrid.py**: An experiment with exhaustive fallback strategies. These concepts were **refined and integrated more professionally** into gradio\_qwen\_enhanced.py.  
* **gradio\_qwen\_optimized.py**: A basic version with conservative settings. This approach is **no longer necessary** due to the superior configuration management in the enhanced version.  
* **gradio\_qwen\_debug.py**: A development file for troubleshooting NPU issues. Its purpose is purely diagnostic and its code is **not suitable for production**.

## **3\. Common Commands**

### **Running Chat Applications**

\# 1\. PRIMARY \- Run the canonical, fully-featured application  
python gradio\_qwen\_enhanced.py

\# \--- LEGACY (For historical comparison or specific debugging ONLY) \---

\# 2\. LEGACY (Stateful API): The first stable stateful implementation.  
python gradio\_qwen\_refined.py

\# 3\. LEGACY (Fallback Logic): An early experiment in advanced fallbacks.  
python gradio\_qwen\_hybrid.py

\# 4\. LEGACY (Basic NPU): A conservative build to bypass old NPUW issues.  
python gradio\_qwen\_optimized.py

\# 5\. LEGACY (Debug): Development version for deep NPU troubleshooting.  
python gradio\_qwen\_debug.py

### **Model and Context Management**

\# Export NPU-compatible model with static shapes  
python export\_qwen\_for\_npu.py \--model Qwen/Qwen2-7B-Instruct \--output qwen2-7b-npu

\# Analyze model compatibility  
python check\_model\_config.py

\# Create consolidated context file  
create\_llm\_context.bat

## **4\. Critical Implementation Patterns**

### **Stateful Chat API (Essential)**

The key architectural insight, confirmed in python\_samples/chat\_sample.py and core\_cpp/pipeline\_stateful.cpp, is that OpenVINO GenAI pipelines are **stateful**. They internally manage conversation history and the KV-cache. Your implementation **must** reflect this for efficiency and correctness.

\# WRONG (Stateless \- causes token limit errors and severe performance issues):  
\# This pattern re-processes the entire conversation with every turn,  
\# quickly exceeding token limits and wasting computational resources.  
\# DO NOT USE THIS PATTERN.  
conversation \= build\_full\_conversation\_history(history \+ \[new\_message\])  
prompt \= tokenizer.apply\_chat\_template(conversation, ...)  
pipe.generate(prompt, ...)

\# RIGHT (Stateful \- efficient and robust, from chat\_sample.py):  
\# On application load or when clearing the chat:  
pipe.start\_chat(SYSTEM\_PROMPT)  \# Initialize with system instructions

\# In the chat function, when the user sends a message:  
pipe.generate(new\_message, ...)  \# Only send the new message \- the pipeline handles the rest

\# On chat clear or application exit:  
pipe.finish\_chat() \# Clears the internal state

### **NPU Configuration Hierarchy**

The NPU requires a specific configuration structure, detailed in core\_cpp/utils.cpp under the update\_npu\_config function.

**Primary NPUW Configuration** (for maximum performance):

\# CRITICAL: Correct NPUW configuration to avoid compilation errors  
npuw\_config \= {  
    "NPU\_USE\_NPUW": "YES",  
    "NPUW\_LLM": "YES",  
    "NPUW\_LLM\_BATCH\_DIM": 0,  
    "NPUW\_LLM\_SEQ\_LEN\_DIM": 1,  
    "NPUW\_LLM\_MAX\_PROMPT\_LEN": 2048,  
    "NPUW\_LLM\_MIN\_RESPONSE\_LEN": 256,  
    \# CRITICAL: Only use "BEST\_PERF" or "FAST\_COMPILE" \- NOT "LATENCY"  
    "NPUW\_LLM\_PREFILL\_HINT": "BEST\_PERF",  
    "NPUW\_LLM\_GENERATE\_HINT": "BEST\_PERF",  
    \# DO NOT include a generic PERFORMANCE\_HINT when using NPUW hints  
}

**Robust Fallback Strategy**:

def try\_load\_pipeline(model\_path, device, config\_priority\_list):  
    """Attempts to load a pipeline with a list of configurations, from highest to lowest priority."""  
    for config\_name, config in config\_priority\_list:  
        try:  
            print(f"Attempting to load pipeline with '{config\_name}' configuration...")  
            return ov\_genai.LLMPipeline(model\_path, device, \*\*config)  
        except Exception as e:  
            print(f"WARN: Failed to load with '{config\_name}': {e}")  
            continue  
    print("ERROR: All pipeline configurations failed.")  
    return None \# Return None if all configurations fail

### **NPU Token Limits and Defensive Design**

The NPU has **hard-coded prompt length limits** (see m\_max\_prompt\_len in pipeline\_stateful.cpp). Sending an oversized input will crash the pipeline. You must protect against this.

**Defensive Pattern**:

\# Protect against NPU prompt limits with user feedback  
MAX\_MESSAGE\_LENGTH \= 2000 \# A safe buffer below the hard limit

if len(message) \> MAX\_MESSAGE\_LENGTH:  
    original\_length \= len(message)  
    \# This is a placeholder; a real implementation should be more intelligent  
    message \= message\[:MAX\_MESSAGE\_LENGTH\]

    \# Inform the user about the truncation  
    truncation\_warning \= (  
        f"⚠️ Your message ({original\_length:,} characters) exceeded the NPU's input limit "  
        f"and was automatically truncated to {len(message):,} characters. Please consider using shorter messages."  
    )  
    history.append({  
        "role": "assistant",  
        "content": truncation\_warning  
    })  
    yield history  \# Show the warning to the user immediately

## **5\. Development Guidelines for OpenVINO GenAI**

### **Guiding Philosophy**

* **Incremental NPU optimization**: Small, successful NPUW configuration changes are better than large, failing rewrites.  
* **Learn from the canonical example**: Study the patterns in gradio\_qwen\_enhanced.py before extending them.  
* **NPU-first pragmatism**: Adapt to Intel NPU hardware constraints, not theoretical ideals.  
* **Clarity through diagnostics**: NPU failures are often cryptic. The solution is to provide excellent, real-time diagnostics in the UI.

### **Process for Adding NPU Features**

1. **Understand NPU constraints**: Study existing working configurations in the codebase.  
2. **Test configuration first**: Validate NPUW settings in isolation before full integration.  
3. **Implement minimal changes**: Modify one configuration parameter at a time.  
4. **Verify NPU compilation**: Ensure each change doesn't break the build.  
5. **Commit working state**: Never commit a non-compiling NPU configuration.

### **When NPU Issues Occur (3-Strike Rule)**

**CRITICAL**: After 3 consecutive failed NPU compilation attempts, **stop and reassess your approach**.

1. **Document the failure**: Note the exact configuration, error message, and driver version.  
2. **Research alternatives**: Check context/qwen\_model\_context/npu\_optimization.py for working patterns and review official Intel NPU documentation.  
3. **Question assumptions**: Is this NPU feature actually supported? Can this be done with a simpler property? Is the stateful API a better solution?  
4. **Try a fallback**: Use a more conservative NPU profile or fall back to CPU for the specific feature.

### **Technical Standards**

* **Architecture**: Use configuration composition over inheritance. Inject model dependencies (don't use globals). Use the stateful API first.  
* **Error Handling**: Fail fast with descriptive messages. A try/except block for NPU compilation should be specific and provide a clear fallback path, not a silent failure.  
* **Decision Framework**: Prioritize choices based on: 1\. NPU Compatibility, 2\. Fallback Safety, 3\. User Transparency, 4\. Maintenance Burden, 5\. Performance Impact.

### **Quality Gates for NPU Features**

A feature is "done" only when:

* \[ \] NPU compilation succeeds with the target configuration.  
* \[ \] The CPU fallback works correctly when the NPU fails.  
* \[ \] The user receives clear feedback about which device is being used.  
* \[ \] Performance metrics are reasonable (e.g., \>10 tokens/sec for NPU).  
* \[ \] Memory usage stays within NPU constraints.  
* \[ \] Long conversations do not cause token limit crashes.

## **6\. Critical Reminders for OpenVINO GenAI**

**NEVER in NPU applications**:

* Use a generic PERFORMANCE\_HINT alongside specific NPUW\_LLM\_\*\_HINT settings (causes compilation conflicts).  
* Set NPUW\_LLM\_GENERATE\_HINT to "LATENCY" (only "BEST\_PERF" or "FAST\_COMPILE" are allowed).  
* Ignore NPU compilation errors—they indicate real configuration problems.  
* Use global pipeline instances in multi-user scenarios.  
* Pass the entire chat history to pipe.generate().

**ALWAYS for production chat apps**:

* Validate NPUW configuration before creating the pipeline.  
* Provide a CPU fallback when NPU compilation fails.  
* Use the stateful API (start\_chat/generate/finish\_chat) for conversation management.  
* Include performance monitoring and user feedback about the active device.  
* Test with various conversation lengths to ensure token limit handling is robust.  
* Filter model-specific special tokens from the streaming output.