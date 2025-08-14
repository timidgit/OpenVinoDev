# Gradio ChatInterface Advanced Patterns
# ======================================
#
# PRIORITY: ⭐⭐⭐⭐⭐ (Essential for system prompts and advanced controls)
#
# This file contains advanced ChatInterface patterns including system prompts,
# additional inputs, and configuration controls. These patterns are crucial for
# production OpenVINO GenAI applications that need fine-grained control.
#
# Key Learning Points:
# - System prompt integration with ChatInterface
# - Additional inputs for generation parameters
# - Multi-parameter streaming functions
# - Advanced ChatInterface configuration options
# - Production-ready parameter controls

import gradio as gr
import time

# =======================================
# PATTERN 1: System Prompt Integration
# =======================================
# Source: demo/chatinterface_system_prompt/run.py
# Essential for controlling LLM behavior

def create_system_prompt_chat():
    """
    ChatInterface with system prompt and token limit controls.
    Perfect for OpenVINO GenAI applications requiring behavior control.
    """
    
    def echo_with_system_prompt(message, history, system_prompt, tokens):
        """
        Streaming function with system prompt and token limit.
        Replace this with OpenVINO GenAI generation.
        """
        response = f"System prompt: {system_prompt}\nMessage: {message}."
        
        # Respect token limit
        max_length = min(len(response), int(tokens))
        
        for i in range(max_length):
            time.sleep(0.05)
            yield response[: i + 1]

    demo = gr.ChatInterface(
        echo_with_system_prompt,
        type="messages",
        additional_inputs=[
            gr.Textbox(
                value="You are a helpful AI assistant powered by OpenVINO.", 
                label="System Prompt",
                info="Controls the AI's behavior and personality"
            ),
            gr.Slider(
                minimum=10, 
                maximum=1000, 
                value=200,
                label="Max Response Tokens",
                info="Limit response length for NPU efficiency"
            ),
        ],
    )
    
    return demo

# =======================================
# PATTERN 2: Advanced Parameter Control
# =======================================
# Enhanced ChatInterface with comprehensive generation controls

def create_advanced_parameter_chat():
    """
    Advanced ChatInterface with full generation parameter control.
    Ideal for OpenVINO GenAI applications requiring fine-tuning.
    """
    
    def generate_with_params(message, history, system_prompt, temperature, top_p, top_k, max_tokens, repetition_penalty):
        """
        Streaming function with comprehensive parameter control.
        This signature matches OpenVINO GenAI GenerationConfig parameters.
        """
        
        # Simulate parameter-aware generation
        # In practice, these parameters would be passed to OpenVINO GenerationConfig
        params_info = f"[T={temperature}, top_p={top_p}, top_k={top_k}, max_tokens={max_tokens}, rep_penalty={repetition_penalty}]"
        response = f"{system_prompt}\n\nUser: {message}\n\nAssistant (with params {params_info}): This is a simulated response that would be generated using the specified parameters."
        
        # Stream with respect to max_tokens
        words = response.split()
        current_response = ""
        
        for i, word in enumerate(words):
            if i >= max_tokens:  # Respect token limit
                break
                
            if i == 0:
                current_response = word
            else:
                current_response += " " + word
            
            time.sleep(0.08)  # Simulate generation with different speeds based on temperature
            yield current_response

    demo = gr.ChatInterface(
        generate_with_params,
        type="messages",
        title="🤖 Advanced OpenVINO GenAI Chat",
        description="Full parameter control for OpenVINO GenAI models",
        additional_inputs=[
            gr.Textbox(
                value="You are an intelligent AI assistant optimized for efficient NPU inference. Provide accurate, helpful responses while being mindful of computational constraints.",
                label="System Prompt",
                lines=3,
                info="Define the AI's role and behavior"
            ),
            gr.Slider(
                minimum=0.1, 
                maximum=2.0, 
                value=0.7,
                step=0.1,
                label="Temperature",
                info="Control randomness (0.1=focused, 2.0=creative)"
            ),
            gr.Slider(
                minimum=0.1, 
                maximum=1.0, 
                value=0.9,
                step=0.05,
                label="Top-p",
                info="Nucleus sampling threshold"
            ),
            gr.Slider(
                minimum=1, 
                maximum=100, 
                value=50,
                step=1,
                label="Top-k", 
                info="Top-k sampling limit"
            ),
            gr.Slider(
                minimum=50, 
                maximum=2048, 
                value=512,
                step=50,
                label="Max Response Tokens",
                info="Maximum tokens to generate (NPU optimized)"
            ),
            gr.Slider(
                minimum=1.0, 
                maximum=2.0, 
                value=1.1,
                step=0.05,
                label="Repetition Penalty",
                info="Penalty for repeating tokens"
            ),
        ],
        additional_inputs_accordion=gr.Accordion("🎛️ Generation Parameters", open=False),
        examples=[
            "Explain quantum computing in simple terms",
            "Write Python code for binary search",
            "Compare NPU vs GPU for AI inference",
            "What are the benefits of model quantization?"
        ],
        cache_examples=False,
    )
    
    return demo

# =====================================
# PATTERN 3: Multi-Modal ChatInterface
# =====================================
# Advanced pattern with file uploads and multi-modal inputs

def create_multimodal_chat():
    """
    Multi-modal ChatInterface supporting text and file inputs.
    Useful for OpenVINO GenAI applications with vision capabilities.
    """
    
    def handle_multimodal_input(message, history, system_prompt, enable_vision):
        """
        Handle both text and potential file inputs.
        Extends to vision-language models with OpenVINO GenAI.
        """
        
        response_prefix = "Vision enabled: " if enable_vision else "Text only: "
        
        # Check if message contains files (multimodal)
        if isinstance(message, dict) and "files" in message:
            file_info = f"[Received {len(message['files'])} file(s)] "
            text_content = message.get("text", "")
            full_response = f"{response_prefix}{file_info}{text_content}"
        else:
            full_response = f"{response_prefix}{message}"
        
        # Stream the response
        for i in range(len(full_response)):
            time.sleep(0.03)
            yield full_response[:i + 1]

    demo = gr.ChatInterface(
        handle_multimodal_input,
        multimodal=True,  # Enable file uploads
        type="messages",
        title="🖼️ Multi-Modal OpenVINO GenAI Chat",
        additional_inputs=[
            gr.Textbox(
                value="You are a multi-modal AI assistant capable of understanding both text and images.",
                label="System Prompt",
                lines=2
            ),
            gr.Checkbox(
                value=False,
                label="Enable Vision Processing",
                info="Use vision-language model for image understanding"
            )
        ],
        examples=[
            {"text": "Hello! How can you help me?"},
            {"text": "Analyze this image for me", "files": ["example.jpg"]},
            {"text": "What can you tell me about AI hardware?"}
        ]
    )
    
    return demo

# =====================================
# PATTERN 4: Session Management Chat
# =====================================
# Advanced session management for stateful OpenVINO GenAI pipelines

def create_session_managed_chat():
    """
    ChatInterface with explicit session management.
    Essential for OpenVINO GenAI stateful pipelines with start_chat/finish_chat.
    """
    
    class SessionState:
        def __init__(self):
            self.session_active = False
            self.session_id = 0
            self.total_tokens = 0
            self.conversation_count = 0
    
    session = SessionState()
    
    def managed_generation(message, history, auto_manage_session, max_conversation_tokens):
        """
        Generation with automatic session management.
        Mimics OpenVINO GenAI stateful pipeline behavior.
        """
        
        if auto_manage_session:
            # Simulate token counting
            estimated_tokens = len(str(history)) + len(message)
            session.total_tokens = estimated_tokens
            
            # Start new session if needed
            if not session.session_active or session.total_tokens > max_conversation_tokens:
                session.session_active = True
                session.session_id += 1
                session.conversation_count = 0
                yield "🔄 New session started (previous session exceeded token limit)\n\n"
            
            session.conversation_count += 1
        
        # Generate response
        response = f"[Session {session.session_id}, Turn {session.conversation_count}] Your message: {message}"
        
        if auto_manage_session:
            response += f"\n\nSession info: {session.total_tokens} tokens used, limit: {max_conversation_tokens}"
        
        # Stream response
        for i in range(len(response)):
            time.sleep(0.02)
            yield response[:i + 1]
    
    def clear_session():
        """Clear session state - equivalent to finish_chat() in OpenVINO GenAI"""
        session.session_active = False
        session.total_tokens = 0
        session.conversation_count = 0
        return None

    demo = gr.ChatInterface(
        managed_generation,
        type="messages",
        title="🔗 Session-Managed OpenVINO GenAI Chat",
        description="Automatic session management for stateful pipelines",
        additional_inputs=[
            gr.Checkbox(
                value=True,
                label="Auto-Manage Sessions",
                info="Automatically start new sessions when token limits exceeded"
            ),
            gr.Slider(
                minimum=500,
                maximum=2048,
                value=1024,
                step=128,
                label="Max Conversation Tokens",
                info="Token limit before starting new session (NPU optimized)"
            )
        ],
        additional_inputs_accordion=gr.Accordion("Session Management", open=True)
    )
    
    # Add custom clear button behavior
    demo.clear_btn.click(clear_session, None, None)
    
    return demo

# =====================================
# INTEGRATION GUIDELINES FOR OPENVINO  
# =====================================

"""
OPENVINO GENAI INTEGRATION GUIDELINES:
=====================================

1. SYSTEM PROMPT INTEGRATION:
   ```python
   # In your streaming function:
   def openvino_generate(message, history, system_prompt, temperature, top_p, max_tokens):
       # Configure generation
       config = ov_genai.GenerationConfig()
       config.temperature = temperature
       config.top_p = top_p  
       config.max_new_tokens = max_tokens
       
       # Build conversation with system prompt
       conversation = [{"role": "system", "content": system_prompt}] + history
       conversation.append({"role": "user", "content": message})
       
       # Generate with OpenVINO
       streamer = YourStreamer(tokenizer)
       pipe.generate(conversation, config, streamer)
       
       for chunk in streamer:
           yield chunk
   ```

2. PARAMETER MAPPING:
   ChatInterface additional_inputs → OpenVINO GenerationConfig:
   - temperature → config.temperature
   - top_p → config.top_p
   - top_k → config.top_k
   - max_tokens → config.max_new_tokens
   - repetition_penalty → config.repetition_penalty

3. SESSION MANAGEMENT:
   For stateful pipelines:
   ```python
   # Start session
   pipe.start_chat(system_prompt)
   
   # Generate (only send current message, not full history)
   pipe.generate(message, config, streamer)
   
   # End session when clearing chat
   pipe.finish_chat()
   ```

4. MULTI-MODAL SUPPORT:
   For vision-language models:
   ```python
   def handle_multimodal(message, history, system_prompt, enable_vision):
       if isinstance(message, dict) and message.get("files"):
           # Handle image files with VLM pipeline
           vlm_pipe.generate(message["text"], message["files"], config, streamer)
       else:
           # Handle text-only with LLM pipeline
           llm_pipe.generate(message, config, streamer)
   ```

RECOMMENDED USAGE:
=================

1. Use PATTERN 1 (System Prompt) for basic OpenVINO GenAI integration
2. Use PATTERN 2 (Advanced Parameters) for production applications 
3. Use PATTERN 3 (Multi-Modal) for vision-language models
4. Use PATTERN 4 (Session Management) for stateful pipelines

The advanced parameter pattern (PATTERN 2) is most suitable for production
OpenVINO GenAI applications as it provides comprehensive control while
maintaining clean separation of concerns.
"""

# Example usage:
if __name__ == "__main__":
    # Choose the pattern most suitable for your OpenVINO GenAI application
    demo = create_advanced_parameter_chat()  # Recommended for production
    demo.launch()