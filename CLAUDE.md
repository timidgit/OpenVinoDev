# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## **Project Overview**

This is a **production-ready, modular Phi-3-mini-128k-instruct chat application** using OpenVINO GenAI with Intel NPU optimization. The project has evolved from a legacy Qwen3 monolithic implementation to a professional modular architecture with comprehensive features including RAG document processing, ReAct agent capabilities, and advanced performance monitoring.

## **Architecture Overview**

### **Unified Entry Point**
- **`main.py`**: Single entry point with comprehensive CLI interface and 4-tier configuration priority
- **Modular `app/` directory**: Production codebase with clear separation of concerns
- **`_context_archive/`**: Legacy implementations preserved for reference (not used in production)

### **Core Module Dependencies**
```
main.py
├── app.config → ConfigurationLoader (4-tier priority system)
├── app.model → deploy_llm_pipeline() (multi-tier NPU fallback)  
├── app.ui → create_enhanced_interface() (Gradio with ChatInterface)
└── app.chat → enhanced_llm_chat() (stateful OpenVINO + RAG)
    ├── app.streamer → EnhancedLLMStreamer (Phi-3 token filtering)
    ├── app.agent → ReAct pattern (optional tool usage)
    └── DocumentRAGSystem (vector search + cross-encoder reranking)
```

## **Common Commands**

### **Development & Testing**
```bash
# Primary application launch
python main.py

# System validation (essential for new environments)
python main.py --validate-only

# Debug mode with comprehensive logging
python main.py --debug

# Device testing and profiling
python main.py --device CPU --debug
python main.py --device NPU --npu-profile balanced --debug

# Model utilities
python check_model_config.py
python export_model_for_npu.py --model microsoft/Phi-3-mini-128k-instruct --output phi3-128k-npu
```

### **CI/CD Integration**
```bash
# Linting (matches CI pipeline)
flake8 app/ --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 app/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Testing (matches CI expectations)  
pytest tests/ -v --cov=app --cov-report=xml --cov-report=html

# Type checking
mypy app/ --ignore-missing-imports --no-strict-optional

# Security scanning
bandit -r app/
safety check
```

### **Installation**
```bash
# Core dependencies (required)
pip install -r requirements.txt

# RAG capabilities (optional but recommended)
pip install langchain faiss-cpu sentence-transformers

# Agent capabilities (optional)
pip install langchain-core langchain-experimental requests python-dateutil
```

## **Critical Architecture Patterns**

### **Configuration Architecture (4-Tier Priority)**
The system implements a sophisticated configuration merger that prevents common configuration bugs:

```python
# Priority: CLI → Environment → JSON → Defaults
config = ConfigurationLoader()
config.load_from_json("config.json")
config.load_from_environment() 
config.override_from_cli(args)
```

**Critical insight**: Environment variables use `MODEL_PATH` (modern) but legacy `QWEN3_MODEL_PATH` is still supported for backward compatibility.

### **NPU Deployment Strategy (Multi-Tier Fallback)**
```python
# app/model.py implements sophisticated deployment logic:
1. Enhanced Phi-3 patterns (app/npu_patterns.py) 
2. Manual NPUW configuration (fallback)
3. Basic OpenVINO properties (minimal)
4. CPU fallback (automatic device switching)
```

**Critical constraint**: NPU requires specific NPUW hints. The codebase has been corrected to use:
- `"NPUW_LLM_PREFILL_HINT": "FAST_COMPILE"` (NOT "LATENCY" - causes compilation errors)
- `"NPUW_LLM_GENERATE_HINT": "BEST_PERF"` (NOT "LATENCY" - unsupported by current drivers)

### **Stateful OpenVINO API Pattern**
OpenVINO GenAI pipelines manage conversation state internally - **never** reconstruct full conversation history:

```python
# CORRECT: Stateful usage
pipe.start_chat(SYSTEM_PROMPT)        # Initialize with system prompt
pipe.generate(new_message, config)    # Send only the new user message
pipe.finish_chat()                    # Clear conversation state

# WRONG: Stateless usage (causes token limit errors)
full_prompt = apply_chat_template(history + [new_message])  
pipe.generate(full_prompt)  # Re-processes entire conversation
```

### **Gradio ChatInterface Compatibility (CRITICAL FIX)**
The UI uses `gr.Chatbot(type='messages')` which requires strict data format compliance. **Recent major fix resolved persistent streaming errors:**

```python
# REQUIRED format: List[Dict[str, str]]
history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

# CRITICAL FIX: UI event handler must yield from generator, not return it
def handle_send(message, history):
    yield from enhanced_llm_chat(message, history, generation_settings)  # CORRECT
    # return enhanced_llm_chat(...)  # WRONG - returns generator object

# All streaming functions must yield properly formatted List[Dict] objects
def enhanced_llm_chat(...) -> Iterator[ChatHistory]:
    yield [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
```

**Debug Pattern**: Use `json.dumps(history, indent=2, default=str)` before every `yield` to verify format compliance.

## **Advanced Features Architecture**

### **RAG System (3-Phase Implementation)**
```python
# Phase 1: Basic text processing (.txt, .md, .py, .js, .html, .css, .json)
# Phase 2: Advanced parsing (unstructured library for .pdf, .docx, .pptx)  
# Phase 3: Cross-encoder reranking (BAAI/bge-reranker-base for quality)

# Usage pattern:
rag_context = rag_system.retrieve_context(query, k=3)
if rag_context:
    augmented_prompt = f"Context: {rag_context}\n\nQuestion: {query}"
```

### **Agent System (ReAct Pattern)**
```python
# Tool detection based on keywords triggers agent vs regular chat
if should_use_agent(message):  # Detects math, dates, analysis requests
    response = agent.process_with_tools(message)
else:
    response = enhanced_llm_chat(message, history)  # Regular streaming chat
```

### **Token Streaming with Phi-3 Filtering**
```python
# app/streamer.py implements Phi-3-specific token filtering:
SPECIAL_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"]
# Filters these during streaming to prevent UI corruption
```

## **Development Guidelines**

### **Model Migration Pattern**
When switching models (current: Phi-3, legacy: Qwen3):
1. **Update `config.json`**: Model path, token limits, cache directory
2. **Update `app/npu_patterns.py`**: Model-specific NPUW configurations  
3. **Update `app/streamer.py`**: Special tokens for new model
4. **Update UI labels**: Model name in `app/ui.py`
5. **Test thoroughly**: `python main.py --validate-only` then `--debug`

### **NPU Development Constraints**
- **Token limits**: NPU has hard-coded prompt length limits (current: 8192 for Phi-3)
- **Defensive programming**: Always validate input length before NPU processing
- **NPUW configuration**: Use supported hint values, avoid generic PERFORMANCE_HINT
- **Memory constraints**: NPU has specific memory limitations requiring careful management

### **Testing Strategy**
```bash
# Component testing
python main.py --validate-only  # System requirements validation
python main.py --debug          # Full system debugging

# Device-specific testing  
python main.py --device CPU     # Test CPU fallback path
python main.py --device NPU     # Test NPU compilation and execution

# Configuration testing
python main.py --model-path "path" --debug  # Test custom model loading
```

### **Error Handling Architecture**
The codebase implements multi-tier error handling:
1. **Input validation**: Security-focused sanitization in `app/chat.py`
2. **Device fallback**: NPU → CPU automatic switching in `app/model.py`
3. **User-friendly errors**: Technical details hidden, actionable messages shown
4. **Graceful degradation**: Features disable cleanly when dependencies missing

## **Recent Critical Fixes & Debugging Insights**

### **Gradio Streaming Format Error (RESOLVED)**
**Problem**: `"Data incompatible with messages format"` errors during streaming
**Root Cause**: UI event handler was returning generator object instead of yielding individual values
**Solution**: Changed `return enhanced_llm_chat(...)` to `yield from enhanced_llm_chat(...)`
**Impact**: Complete resolution of persistent chat interface crashes

### **NPUW Configuration Discovery**
**Problem**: NPU compilation failures with certain hint configurations  
**Root Cause**: Generic `PERFORMANCE_HINT` conflicts with NPUW-specific hints
**Solution**: Use only NPUW-specific hints (`FAST_COMPILE`, `BEST_PERF`) with NPU device

### **History Format Normalization**
**Problem**: Inconsistent message format handling between different Gradio versions
**Solution**: Bulletproof `prepare_chat_input()` function that explicitly rebuilds history on every turn

## **Quality Gates & Production Readiness**

A feature is production-ready when:
- ✅ NPU compilation succeeds with target configuration
- ✅ CPU fallback operates correctly when NPU unavailable
- ✅ User receives clear feedback about active device/configuration
- ✅ Performance metrics meet targets (>15 tokens/sec NPU, >5 tokens/sec CPU)
- ✅ Memory usage stays within device constraints
- ✅ Conversations leverage full 128k context without crashes
- ✅ **Gradio streaming works without format errors**
- ✅ All CI/CD checks pass (linting, testing, type checking, security)

## **Critical Technical Debt & Legacy Naming**

**Note**: Some function/class names retain legacy "qwen3" naming for backward compatibility:
- `enhanced_qwen3_chat()` → processes Phi-3 model
- `EnhancedQwen3Streamer()` → handles Phi-3 token streaming  
- `deploy_qwen3_pipeline()` → deploys Phi-3 pipeline

This naming is intentionally preserved to maintain configuration compatibility and reduce breaking changes during the model transition.

## **Archive System**

The `_context_archive/` directory demonstrates professional technical debt management:
- **Legacy implementations**: Previous monolithic architecture preserved
- **Reference patterns**: Gradio examples and testing frameworks
- **Documentation**: Historical context and enhancement guides  
- **Docker configs**: Containerization examples for different deployment scenarios

This approach enables safe architectural evolution while preserving institutional knowledge.