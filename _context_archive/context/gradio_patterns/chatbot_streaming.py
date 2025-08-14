# Gradio Chatbot Streaming Patterns
# ===================================
# 
# PRIORITY: ⭐⭐⭐⭐⭐ (Essential for OpenVINO GenAI streaming)
# 
# This file contains official Gradio patterns for implementing streaming chatbots.
# These patterns are directly applicable to OpenVINO GenAI applications and provide
# the "official" way to handle streaming responses in Gradio.
#
# Key Learning Points:
# - Character-by-character streaming with proper yielding
# - Clean separation of user input handling and bot response generation
# - Proper use of gr.Blocks with message-type chatbot
# - Queue management for non-blocking UI updates
# - Simple but effective streaming architecture

import gradio as gr
import random
import time

# ================================
# PATTERN 1: Basic Streaming Chat
# ================================
# Source: demo/chatbot_streaming/run.py
# This is the simplest and most reliable streaming pattern

def create_basic_streaming_chat():
    """
    Official Gradio streaming pattern - minimal but complete implementation.
    Perfect template for OpenVINO GenAI integration.
    """
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def user(user_message, history: list):
            """Handle user input and add to history"""
            return "", history + [{"role": "user", "content": user_message}]

        def bot(history: list):
            """Generate streaming bot response"""
            # This would be replaced with OpenVINO GenAI generation
            bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
            history.append({"role": "assistant", "content": ""})
            
            # Character-by-character streaming (replace with token-by-token for LLMs)
            for character in bot_message:
                history[-1]['content'] += character
                time.sleep(0.05)  # Simulate generation time
                yield history

        # Event chain: user input -> bot response
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    return demo

# ======================================
# PATTERN 2: ChatInterface Streaming  
# ======================================
# Source: demo/chatinterface_streaming_echo/run.py
# Higher-level abstraction - easier to implement but less control

def create_chatinterface_streaming():
    """
    ChatInterface with built-in streaming support.
    Good for rapid prototyping of OpenVINO GenAI applications.
    """
    
    def slow_echo(message, history):
        """Streaming function compatible with ChatInterface"""
        for i in range(len(message)):
            time.sleep(0.05)
            yield "You typed: " + message[: i + 1]

    demo = gr.ChatInterface(
        slow_echo,
        type="messages",
        flagging_mode="manual",
        flagging_options=["Like", "Spam", "Inappropriate", "Other"], 
        save_history=True,
    )
    
    return demo

# ====================================
# PATTERN 3: Advanced Streaming Chat
# ====================================
# Enhanced pattern with better error handling and performance monitoring

def create_advanced_streaming_chat():
    """
    Advanced streaming pattern with error handling, performance monitoring,
    and OpenVINO GenAI-specific optimizations.
    """
    
    class StreamingMetrics:
        def __init__(self):
            self.total_requests = 0
            self.avg_response_time = 0
            self.streaming_errors = 0
    
    metrics = StreamingMetrics()
    
    with gr.Blocks() as demo:
        # Enhanced chatbot with better configuration
        chatbot = gr.Chatbot(
            type="messages",
            height=600,
            show_copy_button=True,
            avatar_images=("👤", "🤖")
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Enter your message...",
                scale=4,
                max_lines=3
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")
        
        with gr.Row():
            clear_btn = gr.Button("Clear", variant="secondary")
            metrics_btn = gr.Button("Metrics", variant="secondary")
        
        # Metrics display (collapsible)
        with gr.Row(visible=False) as metrics_row:
            metrics_display = gr.JSON(label="Streaming Performance")

        def user_input_handler(user_message, history: list):
            """Enhanced user input handling with validation"""
            if not user_message.strip():
                return "", history
            
            metrics.total_requests += 1
            return "", history + [{"role": "user", "content": user_message}]

        def streaming_bot_response(history: list):
            """Enhanced bot response with error handling and metrics"""
            if not history:
                return history
            
            start_time = time.time()
            
            try:
                # Simulate OpenVINO GenAI streaming generation
                response = "This is a simulated OpenVINO GenAI response with proper streaming."
                history.append({"role": "assistant", "content": ""})
                
                # Token-by-token streaming (more realistic for LLMs)
                words = response.split()
                for i, word in enumerate(words):
                    if i == 0:
                        history[-1]['content'] = word
                    else:
                        history[-1]['content'] += " " + word
                    
                    time.sleep(0.1)  # Simulate token generation time
                    yield history
                
                # Update metrics
                elapsed = time.time() - start_time
                metrics.avg_response_time = (
                    (metrics.avg_response_time * (metrics.total_requests - 1) + elapsed) 
                    / metrics.total_requests
                )
                
            except Exception as e:
                metrics.streaming_errors += 1
                error_msg = f"❌ Streaming error: {str(e)[:100]}..."
                if history and history[-1]["role"] == "assistant":
                    history[-1]["content"] = error_msg
                else:
                    history.append({"role": "assistant", "content": error_msg})
                yield history

        def get_metrics():
            """Return current streaming metrics"""
            return {
                "total_requests": metrics.total_requests,
                "avg_response_time": round(metrics.avg_response_time, 3),
                "streaming_errors": metrics.streaming_errors,
                "error_rate": round(metrics.streaming_errors / max(metrics.total_requests, 1) * 100, 2)
            }

        def toggle_metrics():
            """Toggle metrics display visibility"""
            return gr.update(visible=True), get_metrics()

        # Event handlers
        msg.submit(
            user_input_handler, [msg, chatbot], [msg, chatbot], queue=False
        ).then(
            streaming_bot_response, chatbot, chatbot
        )
        
        send_btn.click(
            user_input_handler, [msg, chatbot], [msg, chatbot], queue=False  
        ).then(
            streaming_bot_response, chatbot, chatbot
        )
        
        clear_btn.click(lambda: [], None, chatbot, queue=False)
        
        metrics_btn.click(
            toggle_metrics, None, [metrics_row, metrics_display]
        )

    return demo

# =====================================
# INTEGRATION GUIDELINES FOR OPENVINO
# =====================================

"""
OPENVINO GENAI INTEGRATION NOTES:
================================

1. REPLACE SIMULATION WITH REAL GENERATION:
   Replace the simulated streaming in `streaming_bot_response()` with:
   
   ```python
   streamer = YourOpenVINOStreamer(tokenizer)
   pipe.generate(prompt, generation_config, streamer)
   
   for token_text in streamer:
       history[-1]['content'] += token_text
       yield history
   ```

2. TOKEN-LEVEL STREAMING:
   - Use character-by-character for demos/testing
   - Use token-by-token for production LLM applications
   - Consider word-by-word for balanced performance

3. ERROR HANDLING:
   - Wrap OpenVINO generation in try-catch blocks
   - Handle NPU compilation errors gracefully
   - Provide fallback messages for failures

4. PERFORMANCE MONITORING:
   - Track token generation speed (tokens/second)
   - Monitor NPU utilization if available
   - Log compilation and inference times

5. MEMORY MANAGEMENT:
   - Clear conversation history when memory limits reached
   - Implement token counting for NPU constraints
   - Use session management for stateful pipelines

6. UI CONSIDERATIONS:
   - Add generation mode selection (greedy, sampling)
   - Include temperature/top-p controls
   - Show real-time token counts
   - Display device information (NPU/CPU)

USAGE IN YOUR PROJECT:
=====================

The advanced streaming pattern is most suitable for production OpenVINO GenAI
applications. Key benefits:

- Professional error handling
- Built-in performance metrics
- Easy to extend with OpenVINO-specific features
- Compatible with both stateful and stateless pipelines
- Responsive UI with proper queue management

Simply replace the simulated generation with your OpenVINO GenAI streaming
implementation and add NPU-specific configurations.
"""

# Example usage:
if __name__ == "__main__":
    # Choose the pattern most suitable for your needs
    demo = create_advanced_streaming_chat()  # Recommended for production
    demo.launch()