# Qwen3 Special Tokens & Chat Template Guide
# ==========================================
#
# Copyright (c) 2025 sbran
# Licensed under the MIT License
#
# Based on Qwen3 model specifications and tokenizer analysis.
# Incorporates patterns from official Qwen documentation and examples.
#
# PRIORITY: ⭐⭐⭐⭐⭐ (Critical for proper text processing)
#
# This file documents all Qwen3 special tokens, chat template format,
# and proper handling patterns for OpenVINO GenAI applications.

import re
from typing import Dict, List, Optional, Union

# =======================================
# QWEN3 SPECIAL TOKENS MAPPING
# =======================================

QWEN3_SPECIAL_TOKENS = {
    # Core conversation tokens
    151643: "<|endoftext|>",      # BOS/PAD token
    151644: "<|im_start|>",       # Instant Message start
    151645: "<|im_end|>",         # Instant Message end (also EOS)
    
    # Vision and multimodal tokens
    151646: "<|object_ref_start|>",  # Object reference start
    151647: "<|object_ref_end|>",    # Object reference end
    151648: "<|box_start|>",         # Bounding box start
    151649: "<|box_end|>",           # Bounding box end
    151650: "<|quad_start|>",        # Quadrilateral start
    151651: "<|quad_end|>",          # Quadrilateral end
    151652: "<|vision_start|>",      # Vision content start
    151653: "<|vision_end|>",        # Vision content end
    151654: "<|vision_pad|>",        # Vision padding
    151655: "<|image_pad|>",         # Image padding
    151656: "<|video_pad|>",         # Video padding
    
    # Tool calling tokens
    151657: "<tool_call>",           # Tool call start
    151658: "</tool_call>",          # Tool call end
    151665: "<tool_response>",       # Tool response start
    151666: "</tool_response>",      # Tool response end
    
    # Code completion tokens (Fill-in-Middle)
    151659: "<|fim_prefix|>",        # FIM prefix
    151660: "<|fim_middle|>",        # FIM middle
    151661: "<|fim_suffix|>",        # FIM suffix
    151662: "<|fim_pad|>",           # FIM padding
    
    # Repository tokens
    151663: "<|repo_name|>",         # Repository name
    151664: "<|file_sep|>",          # File separator
    
    # Reasoning tokens
    151667: "<think>",               # Thinking start
    151668: "</think>",              # Thinking end
}

# Reverse mapping for token lookup
QWEN3_TOKEN_TO_ID = {token: token_id for token_id, token in QWEN3_SPECIAL_TOKENS.items()}

# Tokens that should be filtered from user-visible output
QWEN3_FILTER_TOKENS = {
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
    "<|image_pad|>", "<|video_pad|>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
    "<think>", "</think>"
}

# =======================================
# QWEN3 CHAT TEMPLATE PATTERNS
# =======================================

class Qwen3ChatTemplate:
    """Qwen3 chat template handler with support for all modes"""
    
    # Basic chat template (no tools)
    BASIC_TEMPLATE = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>"""
    
    # Tool-enabled template
    TOOL_TEMPLATE = """<|im_start|>system
{system_message}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>"""
    
    # Thinking template (with reasoning)
    THINKING_TEMPLATE = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>
{thinking_content}
</think>

{assistant_response}<|im_end|>"""
    
    @classmethod
    def format_basic_chat(cls, system_message: str, user_message: str, assistant_response: str = "") -> str:
        """Format basic chat conversation"""
        return cls.BASIC_TEMPLATE.format(
            system_message=system_message,
            user_message=user_message,
            assistant_response=assistant_response
        )
    
    @classmethod
    def format_with_tools(cls, system_message: str, user_message: str, tools: List[dict], assistant_response: str = "") -> str:
        """Format chat with tool capabilities"""
        import json
        tools_json = json.dumps(tools, indent=2)
        
        return cls.TOOL_TEMPLATE.format(
            system_message=system_message,
            user_message=user_message,
            tools=tools_json,
            assistant_response=assistant_response
        )
    
    @classmethod
    def format_with_thinking(cls, system_message: str, user_message: str, thinking_content: str, assistant_response: str) -> str:
        """Format chat with reasoning/thinking"""
        return cls.THINKING_TEMPLATE.format(
            system_message=system_message,
            user_message=user_message,
            thinking_content=thinking_content,
            assistant_response=assistant_response
        )

# =======================================
# QWEN3 TOKEN FILTERING UTILITIES
# =======================================

class Qwen3TokenFilter:
    """Utilities for filtering and cleaning Qwen3 model output"""
    
    @staticmethod
    def clean_special_tokens(text: str) -> str:
        """Remove Qwen3 special tokens from text output"""
        cleaned = text
        
        # Remove all special tokens
        for token in QWEN3_FILTER_TOKENS:
            cleaned = cleaned.replace(token, "")
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    @staticmethod
    def extract_thinking_content(text: str) -> tuple[str, str]:
        """Extract thinking content from response"""
        thinking_pattern = r'<think>(.*?)</think>'
        thinking_match = re.search(thinking_pattern, text, re.DOTALL)
        
        if thinking_match:
            thinking_content = thinking_match.group(1).strip()
            response_without_thinking = re.sub(thinking_pattern, '', text, flags=re.DOTALL).strip()
            return thinking_content, response_without_thinking
        
        return "", text
    
    @staticmethod
    def extract_tool_calls(text: str) -> tuple[List[dict], str]:
        """Extract tool calls from response"""
        import json
        
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_calls = []
        
        for match in re.finditer(tool_call_pattern, text, re.DOTALL):
            try:
                tool_call_json = match.group(1).strip()
                tool_call = json.loads(tool_call_json)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        # Remove tool calls from main text
        text_without_tools = re.sub(tool_call_pattern, '', text, flags=re.DOTALL).strip()
        
        return tool_calls, text_without_tools
    
    @staticmethod
    def is_special_token_id(token_id: int) -> bool:
        """Check if token ID is a special token"""
        return token_id in QWEN3_SPECIAL_TOKENS
    
    @staticmethod
    def should_display_token(token_id: int) -> bool:
        """Check if token should be displayed to user"""
        if token_id not in QWEN3_SPECIAL_TOKENS:
            return True  # Regular content token
        
        token_text = QWEN3_SPECIAL_TOKENS[token_id]
        return token_text not in QWEN3_FILTER_TOKENS

# =======================================
# QWEN3 STREAMING TOKEN HANDLER
# =======================================

class Qwen3StreamingFilter:
    """Token-level filtering for streaming applications"""
    
    def __init__(self):
        self.accumulated_tokens = []
        self.current_text = ""
        self.in_thinking_mode = False
        self.thinking_buffer = ""
    
    def process_token(self, token_id: int, token_text: str) -> Optional[str]:
        """
        Process a single token and return text to display (if any)
        
        Args:
            token_id: Token ID from model
            token_text: Decoded token text
            
        Returns:
            Text to display to user, or None if token should be filtered
        """
        
        # Handle special tokens
        if token_id in QWEN3_SPECIAL_TOKENS:
            special_token = QWEN3_SPECIAL_TOKENS[token_id]
            
            # Handle thinking mode
            if special_token == "<think>":
                self.in_thinking_mode = True
                return None
            elif special_token == "</think>":
                self.in_thinking_mode = False
                self.thinking_buffer = ""  # Clear thinking buffer
                return None
            
            # Filter other special tokens
            if special_token in QWEN3_FILTER_TOKENS:
                return None
        
        # If in thinking mode, buffer but don't display
        if self.in_thinking_mode:
            self.thinking_buffer += token_text
            return None
        
        # Regular content token - display it
        self.current_text += token_text
        return token_text
    
    def get_thinking_content(self) -> str:
        """Get accumulated thinking content"""
        return self.thinking_buffer
    
    def reset(self):
        """Reset filter state for new conversation"""
        self.accumulated_tokens = []
        self.current_text = ""
        self.in_thinking_mode = False
        self.thinking_buffer = ""

# =======================================
# OPENVINO GENAI INTEGRATION PATTERNS
# =======================================

class Qwen3OpenVINOStreamer:
    """Custom streamer for Qwen3 with proper token filtering"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.filter = Qwen3StreamingFilter()
        self.response_queue = []
    
    def put(self, token_id: int) -> bool:
        """Process token from OpenVINO GenAI pipeline"""
        
        # Decode token
        try:
            token_text = self.tokenizer.decode([token_id])
        except:
            token_text = f"[UNK_{token_id}]"
        
        # Filter and process
        display_text = self.filter.process_token(token_id, token_text)
        
        # Add to queue if should be displayed
        if display_text is not None:
            self.response_queue.append(display_text)
        
        # Continue generation (return False)
        return False
    
    def end(self):
        """Called when generation ends"""
        pass
    
    def get_response_text(self) -> str:
        """Get accumulated response text"""
        return "".join(self.response_queue)
    
    def get_thinking_content(self) -> str:
        """Get thinking content if any"""
        return self.filter.get_thinking_content()

# =======================================
# USAGE EXAMPLES
# =======================================

def example_basic_filtering():
    """Example of basic token filtering"""
    
    # Raw model output with special tokens
    raw_output = "<|im_start|>assistant\nHello! <think>\nThe user is greeting me.\n</think>\nHow can I help you today?<|im_end|>"
    
    # Clean for display
    filter = Qwen3TokenFilter()
    clean_output = filter.clean_special_tokens(raw_output)
    print(f"Cleaned output: {clean_output}")
    
    # Extract thinking content
    thinking, response = filter.extract_thinking_content(raw_output)
    print(f"Thinking: {thinking}")
    print(f"Response: {response}")

def example_chat_formatting():
    """Example of chat template formatting"""
    
    # Basic chat
    system_msg = "You are a helpful AI assistant."
    user_msg = "What is machine learning?"
    
    formatted = Qwen3ChatTemplate.format_basic_chat(system_msg, user_msg)
    print("Basic chat format:")
    print(formatted)
    
    # Chat with tools
    tools = [
        {
            "name": "search_web",
            "description": "Search the internet",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            }
        }
    ]
    
    formatted_tools = Qwen3ChatTemplate.format_with_tools(system_msg, user_msg, tools)
    print("\nTools chat format:")
    print(formatted_tools)

def example_streaming_integration():
    """Example of streaming integration with OpenVINO GenAI"""
    
    import openvino_genai as ov_genai
    
    # Mock tokenizer for example
    class MockTokenizer:
        def decode(self, tokens):
            # Simplified for example
            return f"token_{tokens[0]}"
    
    # Create custom streamer
    tokenizer = MockTokenizer()
    streamer = Qwen3OpenVINOStreamer(tokenizer)
    
    # Simulate token processing
    test_tokens = [
        (151644, "<|im_start|>"),  # Should be filtered
        (12345, "Hello"),          # Should be displayed
        (67890, " world"),         # Should be displayed  
        (151645, "<|im_end|>")     # Should be filtered
    ]
    
    for token_id, expected_text in test_tokens:
        streamer.put(token_id)
    
    streamer.end()
    
    print(f"Final response: {streamer.get_response_text()}")

# =======================================
# INTEGRATION WITH GRADIO APPLICATIONS
# =======================================

def create_qwen3_gradio_streamer(tokenizer):
    """Create Gradio-compatible streamer for Qwen3"""
    
    import queue
    import threading
    
    class Qwen3GradioStreamer(Qwen3OpenVINOStreamer):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            self.text_queue = queue.Queue()
            self.is_generating = True
        
        def put(self, token_id: int) -> bool:
            # Process token
            display_text = self.filter.process_token(token_id, self.tokenizer.decode([token_id]))
            
            if display_text:
                # Add to streaming queue
                self.text_queue.put(display_text)
            
            return False  # Continue generation
        
        def end(self):
            self.is_generating = False
            self.text_queue.put(None)  # End marker
        
        def stream_text(self):
            """Generator for Gradio streaming"""
            accumulated = ""
            while self.is_generating or not self.text_queue.empty():
                try:
                    chunk = self.text_queue.get(timeout=0.1)
                    if chunk is None:  # End marker
                        break
                    accumulated += chunk
                    yield accumulated
                except queue.Empty:
                    continue
    
    return Qwen3GradioStreamer(tokenizer)

# =======================================
# VALIDATION AND TESTING
# =======================================

def validate_qwen3_tokenization():
    """Validate Qwen3 special token handling"""
    
    print("Qwen3 Special Token Validation")
    print("=" * 40)
    
    # Test special token mapping
    for token_id, token_text in list(QWEN3_SPECIAL_TOKENS.items())[:5]:
        print(f"Token ID {token_id}: {token_text}")
        
        # Test filtering
        should_display = not (token_text in QWEN3_FILTER_TOKENS)
        print(f"  Display to user: {should_display}")
    
    print("\nChat Template Validation")
    print("=" * 40)
    
    # Test chat template
    system = "You are helpful."
    user = "Hello!"
    formatted = Qwen3ChatTemplate.format_basic_chat(system, user)
    
    print("Formatted chat:")
    print(formatted[:100] + "..." if len(formatted) > 100 else formatted)
    
    # Test filtering
    filter = Qwen3TokenFilter()
    cleaned = filter.clean_special_tokens(formatted)
    print(f"\nCleaned version: {cleaned}")

if __name__ == "__main__":
    # Run validation
    validate_qwen3_tokenization()
    
    print("\nSpecial tokens loaded:")
    print(f"Total special tokens: {len(QWEN3_SPECIAL_TOKENS)}")
    print(f"Tokens to filter: {len(QWEN3_FILTER_TOKENS)}")
    print(f"Core chat tokens: <|im_start|>, <|im_end|>, <|endoftext|>")