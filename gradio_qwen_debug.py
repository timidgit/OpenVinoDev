import gradio as gr
import openvino_genai as ov_genai
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
from transformers import AutoTokenizer
import time
import queue
from threading import Thread

# --- Strategy 4: Hardware-Level Performance Tuning ---
# Configure the OpenVINO Core for maximum latency performance.
print("Configuring OpenVINO for optimal performance...")
ov_config = {
    # High-level hint (recommended starting point)
    hints.performance_mode: hints.PerformanceMode.LATENCY,
    # Essential for eliminating 'cold start' latency on subsequent runs
    props.cache_dir: ".ovcache"
}

# NPU Pipeline-specific configuration (passed as kwargs)
pipeline_config = {
    "MAX_PROMPT_LEN": 2048,  # Increase from default 1024 for NPU
    "MIN_RESPONSE_LEN": 256   # Ensure minimum response length
}

# NPUW (NPU Wrapper) specific configurations for compilation success
npuw_config = {
    "NPU_USE_NPUW": "YES",           # Enable NPU Wrapper
    "NPUW_LLM": "YES",               # Enable LLM-specific NPU optimizations
    "NPUW_LLM_BATCH_DIM": 0,         # Batch dimension index
    "NPUW_LLM_SEQ_LEN_DIM": 1,       # Sequence length dimension index
    "NPUW_LLM_MAX_PROMPT_LEN": 2048, # Must match MAX_PROMPT_LEN
    "NPUW_LLM_MIN_RESPONSE_LEN": 256, # Must match MIN_RESPONSE_LEN
}

# Advanced NPU optimization properties (minimal - only use core NPUW properties)
advanced_npu_config = {
    # Only use properties that are known to work with NPUW
}

# Try to add NPU-specific settings if available
try:
    # Try the correct NPU properties for 2025
    from openvino.properties import intel_npu
    npu_optimizations = {}
    
    # Try to add available NPU properties
    npu_props_to_try = {
        'turbo': True,
        'compilation_mode_params': 'fast-compile'
    }
    
    for prop_name, prop_value in npu_props_to_try.items():
        try:
            if hasattr(intel_npu, prop_name):
                prop = getattr(intel_npu, prop_name)
                npu_optimizations[prop] = prop_value
                print(f"Added NPU property: {prop_name} = {prop_value}")
            else:
                print(f"NPU property {prop_name} not available in this version")
        except Exception as prop_e:
            print(f"Failed to add NPU property {prop_name}: {prop_e}")
    
    if npu_optimizations:
        ov_config.update(npu_optimizations)
        print(f"NPU-specific optimizations enabled: {len(npu_optimizations)} properties")
    else:
        print("No NPU-specific properties available")
    
except ImportError as ie:
    print(f"Intel NPU properties not available: {ie}")
except Exception as e:
    print(f"Error configuring NPU properties: {e}")

# Add standard performance optimizations
print("Adding standard performance settings...")
ov_config.update({
    props.hint.performance_mode: hints.PerformanceMode.LATENCY,
    props.streams.num: 1,
    props.inference_num_threads: 1
})
print(f"Final device configuration: {list(ov_config.keys())}")

# --- Strategy 1: Prompt Engineering for Conciseness ---
# Define a system prompt to instruct the model on its behavior.
system_prompt = (
    "You are a helpful, direct, and concise AI assistant. "
    "Provide answers immediately without any preamble, self-reflection, or hesitation. "
    "Your responses must be brief and to the point. Do not use think blocks."
)

# --- Strategy 2: Optimizing Generation Parameters ---
# This config uses greedy search for speed and sets strict limits.
generation_config = ov_genai.GenerationConfig()
generation_config.do_sample = False             # Disables sampling for fast, deterministic output.
generation_config.max_new_tokens = 1536         # Hard limit on response length.
generation_config.repetition_penalty = 1.1      # Discourages repetitive loops.

# --- 1. Configuration ---
model_path = r"C:\OpenVinoModels\qwen3-8b-int4-cw-ov"
device = "NPU"  # Can be switched to "CPU" for benchmarking.

# --- 2. Load Model and Tokenizer ---
print(f"Loading model and tokenizer for device: {device}...")
load_start_time = time.time()
try:
    # Try with proper LLMPipeline initialization pattern from samples
    # For NPU device, we need NPUW configuration for compilation success
    if device == "NPU":
        all_config = {**ov_config, **pipeline_config, **npuw_config, **advanced_npu_config}
        print("Using NPU with full NPUW (NPU Wrapper) optimization configuration")
    else:
        all_config = {**ov_config, **pipeline_config}  # Standard config for non-NPU devices
    
    try:
        print(f"Attempting to load with combined config: {list(all_config.keys())}")
        pipe = ov_genai.LLMPipeline(model_path, device, **all_config)
        print("Successfully loaded with full configuration")
    except Exception as config_error:
        print(f"Full config failed ({config_error}), trying NPUW only...")
        try:
            # Try only NPUW config without device properties that might conflict
            pipe = ov_genai.LLMPipeline(model_path, device, **npuw_config, **pipeline_config)
            print("Loaded with NPUW + pipeline configuration only")
        except Exception as npuw_error:
            print(f"NPUW config failed ({npuw_error}), trying device properties with NPUW...")
            try:
                # Try device properties + minimal NPUW config for compilation
                minimal_npuw_config = {**ov_config, **npuw_config}
                pipe = ov_genai.LLMPipeline(model_path, device, **minimal_npuw_config)
                print("Loaded with device properties + NPUW configuration")
            except Exception as mixed_error:
                print(f"Mixed config failed ({mixed_error}), trying device properties only...")
                try:
                    pipe = ov_genai.LLMPipeline(model_path, device, **ov_config)
                    print("Loaded with device properties only")
                except Exception as props_error:
                    print(f"Device properties failed ({props_error}), loading with basic settings...")
                    pipe = ov_genai.LLMPipeline(model_path, device)
                    print("Loaded with basic settings")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Successfully loaded model on {device}")
except Exception as e:
    print(f"Error loading model on {device}: {e}")
    if device == "NPU":
        print("Falling back to CPU...")
        try:
            device = "CPU"
            # Try the same configuration hierarchy for CPU (no NPUW needed)
            all_config_cpu = {**ov_config, **pipeline_config}
            try:
                pipe = ov_genai.LLMPipeline(model_path, device, **all_config_cpu)
            except:
                try:
                    pipe = ov_genai.LLMPipeline(model_path, device, **ov_config)
                except:
                    pipe = ov_genai.LLMPipeline(model_path, device)
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"Successfully loaded model on {device}")
        except Exception as cpu_e:
            print(f"Error loading model on CPU: {cpu_e}")
            exit()
    else:
        exit()

load_end_time = time.time()
print(f"Model and tokenizer loaded successfully in {load_end_time - load_start_time:.2f} seconds.")


# --- Strategy 3: Architecting for Real-Time Interaction with Streaming ---
# This custom streamer bridges the backend generation with the Gradio UI queue.
class GradioStreamer(ov_genai.StreamerBase):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.q = queue.Queue()
        self.accumulated_tokens = []
        self.full_response = ""

    def clean_text(self, text: str) -> str:
        """Remove internal model tokens from text"""
        # Remove common chat template tokens
        tokens_to_remove = [
            '<|im_start|>', '<|im_end|>', 
            '<|system|>', '<|user|>', '<|assistant|>',
            '<|endoftext|>', '<|end|>', '<|start|>',
            '</s>', '<s>', '[INST]', '[/INST]'
        ]
        
        cleaned_text = text
        for token in tokens_to_remove:
            cleaned_text = cleaned_text.replace(token, '')
        
        return cleaned_text

    def put(self, token_id: int) -> bool:
        self.accumulated_tokens.append(token_id)
        # Decode only the new part of the text
        decoded_text = self.tokenizer.decode(self.accumulated_tokens)
        if len(decoded_text) > len(self.full_response):
            new_text = decoded_text[len(self.full_response):]
            # Clean the new text before sending
            cleaned_new_text = self.clean_text(new_text)
            self.full_response = decoded_text
            if cleaned_new_text:  # Only send if there's content after cleaning
                self.q.put(cleaned_new_text)
        return False # Return False to continue generation

    def end(self):
        self.q.put(None) # Signal the end of the stream

    def __iter__(self):
        return self

    def __next__(self):
        item = self.q.get()
        if item is None:
            raise StopIteration
        return item

# --- 3. Define the Generation Function for Gradio ---
def truncate_conversation(conversation, tokenizer, max_tokens=800):
    """Truncate conversation history to fit within NPU token limits"""
    # Start with a more conservative limit for NPU (800 tokens instead of 1500)
    while len(conversation) > 2:  # Keep at least system + current user message
        # Calculate current token count
        test_prompt = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        token_count = len(tokenizer.encode(test_prompt))
        
        print(f"Current conversation token count: {token_count}")
        
        if token_count <= max_tokens:
            break
            
        # Remove oldest user-assistant pair (but keep system message)
        if len(conversation) > 2:
            removed_msg = conversation.pop(1)  # Remove oldest message
            print(f"Removed message: {removed_msg.get('role', 'unknown')} - {len(removed_msg.get('content', ''))} chars")
        else:
            break
    
    return conversation

def chat_function(message: str, history: list):
    try:
        streamer = GradioStreamer(tokenizer)
        
        # Manually construct the conversation with the system prompt
        conversation = [{"role": "system", "content": system_prompt}] + history
        conversation.append({"role": "user", "content": message})
        
        # Truncate conversation if it exceeds token limits (NPU-optimized)
        conversation = truncate_conversation(conversation, tokenizer, max_tokens=800)
        
        prompt = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        
        # Check final prompt length and apply aggressive truncation for NPU
        token_count = len(tokenizer.encode(prompt))
        print(f"Final prompt token count: {token_count}")
        
        if token_count > 1000:  # Much more conservative limit for NPU
            # If still too long, truncate the current user message aggressively
            max_msg_length = 200  # Much shorter limit
            if len(message) > max_msg_length:
                truncated_message = message[:max_msg_length] + "... [truncated]"
                conversation[-1]["content"] = truncated_message
                prompt = tokenizer.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                final_token_count = len(tokenizer.encode(prompt))
                print(f"After message truncation, token count: {final_token_count}")
            
            # Last resort: if still too long, keep only system prompt + current message
            if len(tokenizer.encode(prompt)) > 1000:
                conversation = [conversation[0], conversation[-1]]  # Keep only system + current user
                prompt = tokenizer.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                print(f"Emergency truncation - final token count: {len(tokenizer.encode(prompt))}")
        
        # Add a placeholder for the assistant's response to the history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})

        # Run generation in a separate thread to avoid blocking the UI
        generation_thread = Thread(
            target=pipe.generate,
            args=(prompt, generation_config, streamer)
        )
        generation_thread.start()

        # Yield updates to the UI as they come from the streamer
        for new_text in streamer:
            history[-1]["content"] += new_text
            yield history

    except Exception as e:
        error_msg = str(e)
        print(f"\n!!! An error occurred during generation: {error_msg}")
        
        # Handle specific token limit errors
        if "1024 tokens" in error_msg or "MAX_PROMPT_LEN" in error_msg:
            error_response = "❌ **Token limit exceeded!** The conversation history is too long. Try starting a new chat or use shorter messages."
        else:
            error_response = f"❌ **Error:** {error_msg}"
            
        history.append({"role": "assistant", "content": error_response})
        yield history


# --- 4. Create and Launch the Gradio Interface ---
# The Gradio UI code remains largely the same
with gr.Blocks(theme=gr.themes.Base(), fill_height=True) as demo:
    gr.Markdown("## Qwen-8B OpenVINO Chat (NPU) - Optimized")
    chatbot = gr.Chatbot(label="Conversation", height=600, type='messages',
                         avatar_images=(None, "🤖"))
    with gr.Row():
        msg = gr.Textbox(label="Your Message", placeholder="Type your message here...", scale=7)
        submit_btn = gr.Button("Submit", variant="primary", scale=1)
    clear_btn = gr.Button("Clear Chat", variant="secondary")

    submit_event = msg.submit(chat_function, [msg, chatbot], chatbot)
    submit_event.then(lambda: gr.update(value=""), None, [msg], queue=False)
    button_event = submit_btn.click(chat_function, [msg, chatbot], chatbot)
    button_event.then(lambda: gr.update(value=""), None, [msg], queue=False)
    clear_btn.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

    gr.Examples(examples=["Explain Project Governance in simple terms.", "Write a Python function for Fibonacci."], inputs=msg)

if __name__ == "__main__":
    demo.queue().launch(share=False)