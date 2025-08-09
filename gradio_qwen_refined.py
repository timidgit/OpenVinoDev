#!/usr/bin/env python3
"""
Refined Qwen Chat - Consultant-Inspired Hybrid
===============================================
Combines the best of both worlds with consultant's architectural insights:
- Clean configuration management with intelligent fallback
- Smart token-aware conversation management  
- Professional streaming with controlled decoding
- Performance metrics panel with practical insights
- Production-ready UX for local inference or demos
"""

import gradio as gr
import openvino_genai as ov_genai
import openvino.properties as props
import openvino.properties.hint as hints
from transformers import AutoTokenizer
import time
import queue
import re
from threading import Thread
from typing import Optional, Tuple, Dict, Any

# --- Constants ---
MODEL_PATH = r"C:\OpenVinoModels\qwen3-8b-int4-cw-ov"
DEVICE = "NPU"
CACHE_DIR = r"C:\temp\.ovcache"

# Token management constants
MAX_CONVERSATION_TOKENS = 1024
EMERGENCY_LIMIT = 1500
MAX_MESSAGE_LENGTH = 300

# Global performance tracking
performance_metrics = {
    "total_requests": 0,
    "avg_response_time": 0.0,
    "last_request_tokens": 0,
    "configuration": "unknown",
    "device": "unknown",
    "tokens_per_second": 0.0,
    "session_start": time.time()
}

SYSTEM_PROMPT = (
    "You are a helpful AI assistant powered by OpenVINO on NPU. "
    "Provide accurate, concise responses while being friendly and informative."
)

# --- Configuration Helpers ---
def get_basic_config() -> Dict[str, Any]:
    """Base configuration that works reliably"""
    return {
        hints.performance_mode: hints.PerformanceMode.LATENCY,
        props.cache_dir: CACHE_DIR
    }

def get_enhanced_npu_config() -> Dict[str, Any]:
    """Enhanced NPU configuration with NPUW optimizations"""
    return {
        **get_basic_config(),
        "NPUW_LLM": "YES",
        "NPUW_LLM_MAX_PROMPT_LEN": 4096,  # Increased from 2048 - supports larger initial prompts
        "NPUW_LLM_MIN_RESPONSE_LEN": 128,
        "CACHE_MODE": "OPTIMIZE_SPEED",
        "INFERENCE_PRECISION_HINT": "f16"
    }

def get_cpu_config() -> Dict[str, Any]:
    """Optimized CPU configuration"""
    return {
        **get_basic_config(),
        props.streams.num: 1,
        props.inference_num_threads: 4
    }

# --- Smart Pipeline Loading ---
def load_pipeline_with_fallback(model_path: str, target_device: str) -> Tuple[ov_genai.LLMPipeline, str, str]:
    """
    Load pipeline with intelligent configuration fallback
    Returns: (pipeline, device_used, config_used)
    """
    
    configurations = []
    
    if target_device == "NPU":
        configurations = [
            ("enhanced_npu", target_device, get_enhanced_npu_config()),
            ("basic_npu", target_device, get_basic_config()),
            ("minimal_npu", target_device, {}),
            ("cpu_fallback", "CPU", get_cpu_config())
        ]
    else:
        configurations = [
            ("optimized_cpu", target_device, get_cpu_config()),
            ("basic_cpu", target_device, get_basic_config()),
            ("minimal_cpu", target_device, {})
        ]
    
    for config_name, device, config in configurations:
        try:
            print(f"🔄 Trying {device} with {config_name} configuration...")
            
            if config:
                pipe = ov_genai.LLMPipeline(model_path, device, **config)
            else:
                pipe = ov_genai.LLMPipeline(model_path, device)
                
            print(f"✅ Success: {device} with {config_name}")
            return pipe, device, config_name
            
        except Exception as e:
            print(f"⚠️  {config_name} failed: {e}")
            continue
    
    raise RuntimeError("All configurations failed. Check model path and device drivers.")

# --- Smart Text Processing (Simplified for Stateful API) ---
def smart_truncate_message(text: str, max_length: int) -> str:
    """Intelligently truncate individual messages if too long"""
    if len(text) <= max_length:
        return text
    
    # Try to break at sentence boundaries
    sentences = re.split(r'[.!?]+\s+', text)
    if len(sentences) > 1:
        result = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) + 2 <= max_length * 0.85:
                result.append(sentence)
                current_length += len(sentence) + 2
            else:
                break
        
        if result:
            truncated = '. '.join(result) + '.'
            remaining = text[len(truncated):].strip()
            if len(remaining) > 20:
                truncated += f" [...{remaining[:30].strip()}...]"
            return truncated
    
    # Fallback: word boundary truncation
    truncated = text[:max_length - 50]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.7:
        truncated = truncated[:last_space]
    
    return truncated + f" [...continues for {len(text) - len(truncated)} more chars...]"

# NOTE: manage_conversation_tokens function REMOVED - stateful pipeline handles this automatically!

# --- Enhanced Streaming ---
class RefinedStreamer(ov_genai.StreamerBase):
    """Clean streaming with professional text processing"""
    
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.q = queue.Queue()
        self.tokens = []
        self.text = ""
        self.start_time = time.time()
        
    def clean_output(self, text: str) -> str:
        """Clean model output tokens (simplified - skip_special_tokens handles most cases)"""
        # Most cleaning is handled by skip_special_tokens=True
        # Only do minimal cleanup for edge cases
        special_tokens = [
            '<|im_start|>', '<|im_end|>',
            '<|system|>', '<|user|>', '<|assistant|>'
        ]
        
        cleaned = text
        for token in special_tokens:
            cleaned = cleaned.replace(token, '')
        
        # Normalize whitespace but preserve structure
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def put(self, token_id: int) -> bool:
        self.tokens.append(token_id)
        
        # Decode incrementally
        try:
            decoded = self.tokenizer.decode(self.tokens, skip_special_tokens=True)
        except:
            decoded = self.tokenizer.decode(self.tokens)
        
        if len(decoded) > len(self.text):
            new_text = decoded[len(self.text):]
            self.text = decoded
            
            if new_text.strip():
                self.q.put(new_text)
        
        return False
    
    def end(self):
        # Final cleanup
        if self.text:
            final_text = self.clean_output(self.text)
            if final_text != self.text:
                self.q.put("\n[Text cleaned]")
        
        elapsed = time.time() - self.start_time
        tokens_per_sec = len(self.tokens) / elapsed if elapsed > 0 else 0
        print(f"🚀 Generation: {len(self.tokens)} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
        
        self.q.put(None)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        item = self.q.get()
        if item is None:
            raise StopIteration
        return item

# --- Initialize System ---
print(f"🔧 Loading model from: {MODEL_PATH}")
print(f"🎯 Target device: {DEVICE}")
print(f"💡 Using STATEFUL chat API (following official OpenVINO GenAI pattern)")

try:
    # Load pipeline with smart fallback
    pipe, device_used, config_used = load_pipeline_with_fallback(MODEL_PATH, DEVICE)
    
    # Load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure generation (using correct API pattern)
    generation_config = ov_genai.GenerationConfig()
    generation_config.do_sample = True
    generation_config.temperature = 0.7
    generation_config.top_p = 0.9
    generation_config.top_k = 50
    generation_config.max_new_tokens = 1024
    generation_config.repetition_penalty = 1.1
    # Note: pad_token_id may not be supported in GenerationConfig - handled by tokenizer
    
    # Update global metrics
    performance_metrics["device"] = device_used
    performance_metrics["configuration"] = config_used
    
    print(f"✅ System ready: {device_used} with {config_used} configuration")
    print(f"📚 Following official chat_sample.py pattern with start_chat()/finish_chat()")
    
except Exception as e:
    print(f"❌ Failed to initialize system: {e}")
    exit(1)

# --- Stateful Chat Function (Following Official API Pattern) ---
def stateful_chat_function(message: str, history: list):
    """Stateful chat function using OpenVINO GenAI's built-in conversation management"""
    global performance_metrics
    
    start_time = time.time()
    
    try:
        # Input validation
        if not message.strip():
            yield history
            return
        
        # Handle overly long messages with user-friendly feedback
        original_length = len(message)
        if original_length > MAX_MESSAGE_LENGTH:
            message = smart_truncate_message(message, MAX_MESSAGE_LENGTH)
            print(f"📏 Message truncated: {original_length} → {len(message)} characters")
            
            # Provide immediate feedback to user about truncation
            history.append({
                "role": "assistant", 
                "content": f"⚠️ Your message ({original_length:,} characters) exceeded the NPU limit and was automatically truncated to {len(message)} characters for processing. The response will be based on the truncated content."
            })
            yield history  # Show truncation warning immediately
        
        # Update UI immediately (history is just for display)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        
        # Initialize streaming
        streamer = RefinedStreamer(tokenizer)
        
        # Generate using stateful API - ONLY send the new message!
        def generate():
            try:
                # The pipeline remembers conversation history internally
                # We only need to send the new user message
                pipe.generate(message, generation_config, streamer)
            except Exception as e:
                print(f"❌ Generation error: {e}")
                streamer.q.put(f"❌ Generation error: {str(e)[:100]}...")
                streamer.q.put(None)
        
        generation_thread = Thread(target=generate)
        generation_thread.start()
        
        # Stream to UI
        for chunk in streamer:
            if chunk:  # Only add non-empty chunks
                history[-1]["content"] += chunk
                yield history
        
        generation_thread.join(timeout=30.0)
        
        # Update performance metrics (simplified)
        elapsed = time.time() - start_time
        output_tokens = len(tokenizer.encode(history[-1]["content"]) if history[-1]["content"] else [])
        input_tokens = len(tokenizer.encode(message))
        
        performance_metrics["total_requests"] += 1
        performance_metrics["avg_response_time"] = (
            performance_metrics["avg_response_time"] * (performance_metrics["total_requests"] - 1) + elapsed
        ) / performance_metrics["total_requests"]
        performance_metrics["last_request_tokens"] = input_tokens
        performance_metrics["tokens_per_second"] = output_tokens / elapsed if elapsed > 0 else 0
        
        print(f"📊 Stateful Request: {elapsed:.2f}s, {input_tokens}→{output_tokens} tokens ({performance_metrics['tokens_per_second']:.1f} tok/s)")
        
    except Exception as e:
        error_message = f"❌ Chat error: {str(e)[:150]}..."
        
        # Handle different error types
        if "token" in str(e).lower() or "length" in str(e).lower():
            error_message = "⚠️ Message too long or conversation exceeded limits. Try starting a new chat."
        elif "memory" in str(e).lower():
            error_message = "⚠️ Memory limit reached. Please start a new conversation."
            
        if not history or history[-1]["role"] != "assistant":
            history.append({"role": "assistant", "content": error_message})
        else:
            history[-1]["content"] = error_message
            
        print(f"❌ Stateful chat error: {e}")
        yield history

# --- Gradio Interface ---
def create_interface():
    """Create refined Gradio interface"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Refined Qwen Chat",
        css="""
        .gradio-container { max-width: 1200px; margin: auto; }
        .chatbot { height: 600px; }
        .metrics-panel { background: #f8f9fa; padding: 12px; border-radius: 6px; }
        """,
    ) as demo:
        
        # Header
        gr.Markdown(f"""
        # 🤖 Refined Qwen Chat (Stateful)
        
        **Device**: {device_used} | **Config**: {config_used} | **Model**: Qwen3-8B INT4  
        **Mode**: Stateful conversation management (following OpenVINO GenAI best practices)
        """)
        
        # Main chat
        chatbot = gr.Chatbot(
            label="Conversation",
            height=600,
            type='messages',
            avatar_images=(None, "🤖"),
            show_copy_button=True
        )
        
        # Input
        with gr.Row():
            msg = gr.Textbox(
                placeholder=f"Chat with Qwen on {device_used}...",
                scale=8,
                max_lines=3,
                show_label=False
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Controls
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            metrics_btn = gr.Button("📊 Metrics", variant="secondary")
            info_btn = gr.Button("ℹ️ Info", variant="secondary")
        
        # Metrics panel (collapsible)
        with gr.Row(visible=False) as metrics_row:
            with gr.Column(elem_classes=["metrics-panel"]):
                metrics_display = gr.JSON(label="Performance Metrics")
        
        # Event handlers with stateful session management
        msg.submit(stateful_chat_function, [msg, chatbot], chatbot).then(
            lambda: gr.update(value=""), None, [msg]
        )
        
        send_btn.click(stateful_chat_function, [msg, chatbot], chatbot).then(
            lambda: gr.update(value=""), None, [msg]
        )
        
        # Clear chat: finish old session and start new one
        clear_btn.click(
            lambda: ([], ""), None, [chatbot, msg]
        ).then(
            lambda: pipe.finish_chat(), None, None  # End current session
        ).then(
            lambda: pipe.start_chat(SYSTEM_PROMPT), None, None   # Start new session with system prompt
        )
        
        def show_metrics():
            session_time = time.time() - performance_metrics["session_start"]
            metrics = {
                **performance_metrics,
                "session_duration": f"{session_time:.1f}s"
            }
            return gr.update(value=metrics, visible=True), gr.update(visible=True)
        
        metrics_btn.click(show_metrics, None, [metrics_display, metrics_row])
        
        def show_info():
            info = f"""
            ## System Information
            - **Device**: {device_used}
            - **Configuration**: {config_used}
            - **Model Path**: `{MODEL_PATH}`
            - **Cache Directory**: `{CACHE_DIR}`
            - **API Mode**: Stateful (using start_chat/finish_chat)
            - **Conversation Management**: Automatic (handled by OpenVINO GenAI)
            - **Generation Settings**: Temperature=0.7, Top-p=0.9, Top-k=50
            - **Key Benefit**: No manual token counting - pipeline manages KV-cache internally
            """
            gr.Info(info)
        
        info_btn.click(show_info, None, None)
        
        # Examples
        gr.Examples([
            "Explain quantum computing simply",
            "Write Python code for a binary tree",
            "What's new in AI hardware?",
            "Compare CPU vs GPU vs NPU for AI"
        ], inputs=msg)
        
        # Initialize stateful chat when interface loads (MUST be inside gr.Blocks context)
        demo.load(lambda: pipe.start_chat(SYSTEM_PROMPT), None, None)
    
    return demo

# --- Launch ---
if __name__ == "__main__":
    print("🌐 Starting Refined Qwen Chat Interface...")
    print("✨ Features: Stateful chat sessions, smart fallback, performance tracking")
    print("🎯 Key Improvement: Uses OpenVINO GenAI's built-in conversation management")
    
    demo = create_interface()
    demo.queue(max_size=10).launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )