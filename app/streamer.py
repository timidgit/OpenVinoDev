"""
Token Streaming and Filtering
============================

Enhanced streaming implementation with Phi-3-specific token filtering,
performance monitoring, and robust error handling.
"""

import time
import queue
from typing import Optional
import openvino_genai as ov_genai
from transformers import AutoTokenizer

# Import enhanced context patterns
import os
import sys
context_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "context")
sys.path.insert(0, context_path)

# Import Qwen3-specific filtering
try:
    from qwen3_model_context.special_tokens import (
        Qwen3StreamingFilter,
        QWEN3_SPECIAL_TOKENS
    )
    ENHANCED_CONTEXT_AVAILABLE = True
    print("✅ Enhanced Phi-3 token filtering loaded")
except ImportError:
    print("⚠️ Enhanced token filtering not available - using fallback")
    ENHANCED_CONTEXT_AVAILABLE = False


class EnhancedLLMStreamer(ov_genai.StreamerBase):
    """
    Production-ready streamer with Phi-3-specific optimizations:
    - Proper special token filtering (26+ tokens)
    - Performance monitoring
    - Robust error handling
    - Token-level streaming control
    """
    
    def __init__(self, tokenizer: AutoTokenizer, metrics_callback=None):
        """
        Initialize enhanced streamer.
        
        Args:
            tokenizer: HuggingFace tokenizer for decoding
            metrics_callback: Optional callback to update global metrics
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.text_queue = queue.Queue()
        self.accumulated_tokens = []
        self.current_text = ""
        self.start_time = time.time()
        self.first_token_time = None
        self.tokens_generated = 0
        self.metrics_callback = metrics_callback
        
        # Initialize Qwen3-specific filtering
        if ENHANCED_CONTEXT_AVAILABLE:
            self.token_filter = Qwen3StreamingFilter()
            print("✅ Using enhanced Qwen3 token filtering")
        else:
            self.token_filter = None
            print("⚠️ Using basic token filtering")
    
    def put(self, token_id: int) -> bool:
        """Process token with Qwen3-specific filtering"""
        self.accumulated_tokens.append(token_id)
        self.tokens_generated += 1
        
        # Record first token latency
        if self.first_token_time is None:
            self.first_token_time = time.time()
        
        try:
            # Decode with special token handling
            if ENHANCED_CONTEXT_AVAILABLE and self.token_filter:
                # Use enhanced filtering
                try:
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                except:
                    token_text = f"[UNK_{token_id}]"
                
                # Process through Qwen3 filter
                display_text = self.token_filter.process_token(token_id, token_text)
                
                if display_text is not None:
                    self.current_text += display_text
                    self.text_queue.put(display_text)
                else:
                    # Token was filtered (special token)
                    if self.metrics_callback:
                        self.metrics_callback("special_tokens_filtered", 1)
                    
            else:
                # Fallback filtering
                try:
                    # Decode incrementally
                    full_text = self.tokenizer.decode(self.accumulated_tokens, skip_special_tokens=True)
                    
                    if len(full_text) > len(self.current_text):
                        new_text = full_text[len(self.current_text):]
                        self.current_text = full_text
                        
                        # Basic special token filtering
                        if not self._is_special_token_text(new_text):
                            self.text_queue.put(new_text)
                except Exception as e:
                    print(f"⚠️ Decoding error: {e}")
                    return False
                    
        except Exception as e:
            print(f"❌ Token processing error: {e}")
            return False
        
        return False  # Continue generation
    
    def _is_special_token_text(self, text: str) -> bool:
        """Basic special token detection for fallback"""
        special_patterns = [
            '<|im_start|>', '<|im_end|>', '<|endoftext|>',
            '<think>', '</think>', '<tool_call>', '</tool_call>',
            '<|system|>', '<|user|>', '<|assistant|>'
        ]
        
        for pattern in special_patterns:
            if pattern in text:
                if self.metrics_callback:
                    self.metrics_callback("special_tokens_filtered", 1)
                return True
        
        return False
    
    def end(self):
        """Finalize streaming and calculate performance metrics"""
        # Calculate performance metrics
        total_time = time.time() - self.start_time
        first_token_latency = (self.first_token_time - self.start_time) if self.first_token_time else 0
        tokens_per_second = self.tokens_generated / total_time if total_time > 0 else 0
        
        # Update global metrics via callback
        if self.metrics_callback:
            self.metrics_callback("first_token_latency", first_token_latency)
            self.metrics_callback("tokens_per_second", tokens_per_second)
            self.metrics_callback("total_tokens_generated", self.tokens_generated)
        
        # Log performance
        print(f"🚀 Generation complete: {self.tokens_generated} tokens in {total_time:.2f}s")
        print(f"   First token: {first_token_latency:.3f}s, Speed: {tokens_per_second:.1f} tok/s")
        
        if ENHANCED_CONTEXT_AVAILABLE and self.token_filter:
            thinking_content = self.token_filter.get_thinking_content()
            if thinking_content.strip():
                print(f"🧠 Model thinking: {thinking_content[:100]}...")
        
        # Signal end of generation
        self.text_queue.put(None)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        item = self.text_queue.get()
        if item is None:
            raise StopIteration
        return item


class StreamingMetrics:
    """Simple metrics tracking for streaming performance"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0
        self.avg_first_token_latency = 0.0
        self.avg_tokens_per_second = 0.0
        self.total_tokens_generated = 0
        self.special_tokens_filtered = 0
        self.session_start_time = time.time()
    
    def update_metric(self, metric_name: str, value):
        """Update a specific metric"""
        if metric_name == "special_tokens_filtered":
            self.special_tokens_filtered += value
        elif metric_name == "total_tokens_generated":
            self.total_tokens_generated += value
        elif metric_name == "first_token_latency":
            # Calculate running average
            if self.successful_requests > 0:
                self.avg_first_token_latency = (
                    (self.avg_first_token_latency * (self.successful_requests - 1) + value)
                    / self.successful_requests
                )
            else:
                self.avg_first_token_latency = value
        elif metric_name == "tokens_per_second":
            # Calculate running average
            if self.successful_requests > 0:
                self.avg_tokens_per_second = (
                    (self.avg_tokens_per_second * (self.successful_requests - 1) + value)
                    / self.successful_requests
                )
            else:
                self.avg_tokens_per_second = value
    
    def start_request(self):
        """Mark the start of a new request"""
        self.total_requests += 1
    
    def end_request(self, success: bool, response_time: float):
        """Mark the end of a request"""
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.total_requests - 1) + response_time)
            / self.total_requests
        )
    
    def get_summary(self) -> dict:
        """Get metrics summary"""
        session_duration = time.time() - self.session_start_time
        success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100
        
        return {
            "Session Duration": f"{session_duration:.1f}s",
            "Total Requests": self.total_requests,
            "Success Rate": f"{success_rate:.1f}%",
            "Avg Response Time": f"{self.avg_response_time:.2f}s",
            "Avg First Token": f"{self.avg_first_token_latency:.3f}s",
            "Avg Tokens/Second": f"{self.avg_tokens_per_second:.1f}",
            "Total Tokens Generated": self.total_tokens_generated,
            "Special Tokens Filtered": self.special_tokens_filtered,
        }


# Global metrics instance
streaming_metrics = StreamingMetrics()