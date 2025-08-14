# HuggingFace + Gradio Integration Patterns for OpenVINO GenAI
# =============================================================
#
# PRIORITY: ⭐⭐⭐⭐ (Important for model integration best practices)
#
# This file contains official HuggingFace integration patterns that demonstrate
# proper model loading, tokenizer usage, and chat template application. These
# patterns are directly applicable to OpenVINO GenAI applications.
#
# Key Learning Points:
# - Proper tokenizer integration and configuration
# - Chat template application for conversation formatting
# - Model loading and device management patterns
# - Token decoding and post-processing
# - Error handling for model operations

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# =======================================
# PATTERN 1: Basic HuggingFace Integration
# =======================================
# Source: demo/llm_hf_transformers/run.py
# Shows the standard pattern for integrating transformers models

def create_basic_hf_integration():
    """
    Basic HuggingFace transformers integration pattern.
    Demonstrates concepts directly applicable to OpenVINO GenAI.
    """
    
    # Model loading pattern (replace with OpenVINO GenAI equivalent)
    checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
    device = "cpu"  # "cuda" or "cpu" - equivalent to OpenVINO device selection
    
    print(f"Loading model: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    
    def predict(message, history):
        """
        Core prediction function showing HuggingFace patterns.
        This maps directly to OpenVINO GenAI workflows.
        """
        
        # 1. Build conversation history (same for OpenVINO GenAI)
        history.append({"role": "user", "content": message})
        
        # 2. Apply chat template (OpenVINO GenAI uses same tokenizer.apply_chat_template)
        input_text = tokenizer.apply_chat_template(history, tokenize=False)
        
        # 3. Tokenize input (OpenVINO GenAI handles this internally)
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        # 4. Generate response (equivalent to pipe.generate in OpenVINO GenAI)
        outputs = model.generate(
            inputs, 
            max_new_tokens=100,
            temperature=0.2,
            top_p=0.9,
            do_sample=True
        )
        
        # 5. Decode response (similar post-processing needed for OpenVINO GenAI)
        decoded = tokenizer.decode(outputs[0])
        
        # 6. Extract assistant response from full conversation
        response = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
        
        return response

    demo = gr.ChatInterface(predict, type="messages")
    return demo

# =======================================
# PATTERN 2: Advanced Tokenizer Usage
# =======================================
# Enhanced tokenizer patterns for OpenVINO GenAI integration

def create_advanced_tokenizer_demo():
    """
    Advanced tokenizer usage patterns applicable to OpenVINO GenAI.
    Shows proper handling of special tokens, padding, and conversation formatting.
    """
    
    # Simulate OpenVINO GenAI model setup
    model_path = "microsoft/DialoGPT-medium"  # Example model
    
    # Load tokenizer with proper configuration
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Configure tokenizer for chat applications (same for OpenVINO GenAI)
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Add chat template if not present (common need for OpenVINO GenAI)
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}{% elif message['role'] == 'system' %}{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"""
    
    def advanced_predict(message, history, system_prompt, max_tokens, temperature):
        """
        Advanced prediction with comprehensive tokenizer usage.
        Shows patterns essential for OpenVINO GenAI integration.
        """
        
        # Build full conversation with system prompt
        conversation = []
        if system_prompt.strip():
            conversation.append({"role": "system", "content": system_prompt})
        
        # Add history and current message
        conversation.extend(history)
        conversation.append({"role": "user", "content": message})
        
        try:
            # Apply chat template with proper settings
            formatted_prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize with attention to special tokens
            inputs = tokenizer.encode(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048  # Adjust based on model limits
            )
            
            # Simulate generation (replace with OpenVINO GenAI pipe.generate)
            # This shows the parameters that map to OpenVINO GenerationConfig
            simulated_response = f"[Simulated] Response to '{message}' with temp={temperature}, max_tokens={max_tokens}"
            
            # Token counting for monitoring (important for OpenVINO GenAI)
            input_tokens = len(inputs[0])
            output_tokens = len(tokenizer.encode(simulated_response))
            
            response_with_stats = f"{simulated_response}\n\n📊 Tokens: {input_tokens} input → {output_tokens} output"
            
            return response_with_stats
            
        except Exception as e:
            return f"❌ Error during generation: {str(e)}"
    
    demo = gr.ChatInterface(
        advanced_predict,
        type="messages",
        title="🔧 Advanced Tokenizer Integration",
        additional_inputs=[
            gr.Textbox(
                value="You are a helpful AI assistant.",
                label="System Prompt",
                lines=2
            ),
            gr.Slider(
                minimum=50,
                maximum=512,
                value=150,
                label="Max Tokens"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            )
        ]
    )
    
    return demo

# =======================================
# PATTERN 3: Streaming Integration
# =======================================
# Streaming patterns compatible with both HuggingFace and OpenVINO GenAI

def create_streaming_integration_demo():
    """
    Streaming integration pattern showing token-by-token generation.
    Demonstrates streaming concepts applicable to OpenVINO GenAI.
    """
    
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure for proper streaming
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    class StreamingTokenizer:
        """
        Token-level streaming class.
        Similar concept to OpenVINO GenAI StreamerBase.
        """
        
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.accumulated_tokens = []
            self.accumulated_text = ""
        
        def add_token(self, token_id):
            """Add token and return incremental decoded text"""
            self.accumulated_tokens.append(token_id)
            
            # Decode accumulated tokens
            try:
                new_text = self.tokenizer.decode(
                    self.accumulated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # Return only the new portion
                if len(new_text) > len(self.accumulated_text):
                    new_portion = new_text[len(self.accumulated_text):]
                    self.accumulated_text = new_text
                    return new_portion
                
            except Exception:
                # Fallback for partial tokens
                return ""
            
            return ""
        
        def reset(self):
            """Reset accumulator"""
            self.accumulated_tokens = []
            self.accumulated_text = ""
    
    def streaming_predict(message, history):
        """
        Streaming prediction showing token-by-token processing.
        This pattern translates directly to OpenVINO GenAI streaming.
        """
        
        # Prepare conversation
        conversation = history + [{"role": "user", "content": message}]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Simulate streaming token generation
        # In OpenVINO GenAI, this would be: pipe.generate(prompt, config, streamer)
        simulated_response = f"This is a streaming response to your message: '{message}'. Each word appears progressively to demonstrate token-level streaming capabilities."
        
        words = simulated_response.split()
        current_response = ""
        
        for i, word in enumerate(words):
            if i == 0:
                current_response = word
            else:
                current_response += " " + word
            
            time.sleep(0.1)  # Simulate token generation time
            yield current_response
    
    def streaming_chat(message, history):
        """
        Chat interface compatible streaming function.
        Maps to OpenVINO GenAI streaming patterns.
        """
        response_generator = streaming_predict(message, history)
        
        for partial_response in response_generator:
            yield partial_response
    
    demo = gr.ChatInterface(
        streaming_chat,
        type="messages",
        title="🌊 Streaming Integration Demo",
        description="Token-by-token streaming compatible with OpenVINO GenAI patterns"
    )
    
    return demo

# =======================================
# PATTERN 4: Error Handling & Fallbacks
# =======================================
# Robust error handling patterns for production use

def create_robust_integration_demo():
    """
    Robust integration with comprehensive error handling.
    Essential patterns for production OpenVINO GenAI applications.
    """
    
    class RobustModelHandler:
        """Model handler with comprehensive error handling"""
        
        def __init__(self, model_path: str):
            self.model_path = model_path
            self.tokenizer = None
            self.model = None
            self.loaded = False
            self.load_errors = []
        
        def load_model(self):
            """Load model with proper error handling"""
            try:
                print(f"🔄 Loading tokenizer from {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
                # Configure tokenizer
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                print(f"🔄 Loading model from {self.model_path}")
                # In OpenVINO GenAI: pipe = ov_genai.LLMPipeline(model_path, device)
                self.model = "simulated_openvino_pipeline"  # Placeholder
                
                self.loaded = True
                print("✅ Model loaded successfully")
                
            except Exception as e:
                error_msg = f"Failed to load model: {str(e)}"
                self.load_errors.append(error_msg)
                print(f"❌ {error_msg}")
                self.loaded = False
        
        def generate_with_fallback(self, message: str, history: list, **kwargs):
            """Generate response with fallback strategies"""
            
            if not self.loaded:
                return "❌ Model not loaded. Please check the model path and try again."
            
            try:
                # Build conversation
                conversation = history + [{"role": "user", "content": message}]
                
                # Apply chat template with error handling
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    # Fallback: manual template
                    prompt = f"User: {message}\nAssistant:"
                    print(f"⚠️ Chat template failed, using fallback: {e}")
                
                # Simulate generation with error handling
                # In OpenVINO GenAI: result = pipe.generate(prompt, generation_config, streamer)
                try:
                    response = f"Generated response for: '{message}' (using robust error handling)"
                    
                    # Simulate potential generation errors
                    if len(message) > 1000:  # Simulate token limit error
                        raise ValueError("Input too long")
                    
                    return response
                    
                except ValueError as e:
                    if "too long" in str(e):
                        # Truncate and retry
                        truncated_message = message[:500] + "... [truncated]"
                        return f"⚠️ Input truncated due to length. Response to: '{truncated_message}'"
                    else:
                        raise e
                
            except Exception as e:
                # Final fallback
                error_response = f"❌ Generation failed: {str(e)[:100]}..."
                
                # Suggest actions based on error type
                if "memory" in str(e).lower():
                    error_response += "\n💡 Try a shorter message or restart the application."
                elif "timeout" in str(e).lower():
                    error_response += "\n💡 Request timed out. Please try again."
                elif "compilation" in str(e).lower():
                    error_response += "\n💡 Model compilation failed. Check device compatibility."
                
                return error_response
    
    # Initialize robust handler
    handler = RobustModelHandler("microsoft/DialoGPT-small")
    handler.load_model()
    
    def robust_predict(message, history, enable_fallback, max_retries):
        """Prediction with configurable error handling"""
        
        retries = 0
        while retries < max_retries:
            try:
                result = handler.generate_with_fallback(message, history)
                
                if result.startswith("❌") and enable_fallback and retries < max_retries - 1:
                    retries += 1
                    time.sleep(1)  # Brief delay before retry
                    continue
                
                return result
                
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    return f"❌ Failed after {max_retries} retries: {str(e)}"
                time.sleep(1)
        
        return "❌ Unexpected error in retry logic"
    
    demo = gr.ChatInterface(
        robust_predict,
        type="messages",
        title="🛡️ Robust Integration with Error Handling",
        description="Production-ready error handling and fallback strategies",
        additional_inputs=[
            gr.Checkbox(
                value=True,
                label="Enable Automatic Fallbacks",
                info="Automatically handle common errors"
            ),
            gr.Slider(
                minimum=1,
                maximum=5,
                value=2,
                step=1,
                label="Max Retries",
                info="Number of retry attempts on failure"
            )
        ]
    )
    
    return demo

# =====================================
# INTEGRATION GUIDELINES FOR OPENVINO
# =====================================

"""
OPENVINO GENAI INTEGRATION GUIDELINES:
=====================================

1. TOKENIZER INTEGRATION:
   ```python
   # Load tokenizer (same as HuggingFace)
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   
   # Configure for OpenVINO GenAI
   if tokenizer.pad_token_id is None:
       tokenizer.pad_token_id = tokenizer.eos_token_id
   
   # Load OpenVINO GenAI pipeline
   pipe = ov_genai.LLMPipeline(model_path, device)
   ```

2. CHAT TEMPLATE APPLICATION:
   ```python
   # Build conversation (same as HuggingFace)
   conversation = [{"role": "system", "content": system_prompt}] + history
   conversation.append({"role": "user", "content": message})
   
   # Apply chat template (same as HuggingFace)
   prompt = tokenizer.apply_chat_template(
       conversation,
       tokenize=False,
       add_generation_prompt=True
   )
   
   # Generate with OpenVINO GenAI
   response = pipe.generate(prompt, generation_config)
   ```

3. STREAMING INTEGRATION:
   ```python
   # Create custom streamer (extends ov_genai.StreamerBase)
   class OpenVINOStreamer(ov_genai.StreamerBase):
       def __init__(self, tokenizer):
           super().__init__()
           self.tokenizer = tokenizer
           self.tokens = []
       
       def put(self, token_id):
           self.tokens.append(token_id)
           # Decode incrementally
           text = self.tokenizer.decode(self.tokens, skip_special_tokens=True)
           return False  # Continue generation
   ```

4. ERROR HANDLING PATTERNS:
   ```python
   def openvino_generate_with_fallback(message, history):
       try:
           # Try main generation
           return pipe.generate(prompt, generation_config)
       except ov_genai.CompilationError:
           # Handle NPU compilation issues
           return generate_with_cpu_fallback(prompt)
       except ov_genai.TokenLimitError:
           # Handle token limit exceeded
           truncated_prompt = truncate_conversation(prompt)
           return pipe.generate(truncated_prompt, generation_config)
   ```

5. DEVICE MANAGEMENT:
   ```python
   # Device selection (similar to HuggingFace device handling)
   devices = ["NPU", "CPU", "GPU"]
   
   for device in devices:
       try:
           pipe = ov_genai.LLMPipeline(model_path, device)
           print(f"✅ Loaded on {device}")
           break
       except Exception as e:
           print(f"❌ {device} failed: {e}")
           continue
   ```

KEY DIFFERENCES FROM HUGGINGFACE:
===============================

1. MODEL LOADING:
   - HuggingFace: AutoModelForCausalLM.from_pretrained()
   - OpenVINO GenAI: ov_genai.LLMPipeline(model_path, device)

2. GENERATION:
   - HuggingFace: model.generate(inputs, **generation_kwargs)
   - OpenVINO GenAI: pipe.generate(prompt, generation_config, streamer)

3. STREAMING:
   - HuggingFace: Custom implementation required
   - OpenVINO GenAI: Built-in StreamerBase class

4. DEVICE HANDLING:
   - HuggingFace: .to(device)
   - OpenVINO GenAI: Device specified at pipeline creation

RECOMMENDED USAGE:
=================

1. Use PATTERN 1 for understanding basic integration concepts
2. Use PATTERN 2 for advanced tokenizer handling
3. Use PATTERN 3 for streaming implementations
4. Use PATTERN 4 for production error handling

The robust integration pattern (PATTERN 4) is essential for production
OpenVINO GenAI applications where reliability is critical.
"""

# Example usage:
if __name__ == "__main__":
    # Choose the pattern most suitable for your integration needs
    demo = create_robust_integration_demo()  # Recommended for production
    demo.launch()