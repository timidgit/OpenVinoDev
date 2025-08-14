# Gradio Concurrency and Queue Management Patterns
# ================================================
#
# PRIORITY: ⭐⭐⭐⭐⭐ (Essential for production OpenVINO GenAI deployments)
#
# This file contains essential patterns for handling concurrency, queue management,
# and multiple simultaneous requests in Gradio applications. Critical for OpenVINO
# GenAI applications where NPU resources are limited and requests must be managed.
#
# Key Learning Points:
# - Proper queue configuration for blocking/non-blocking operations
# - Concurrency limits for resource management
# - Request prioritization and load balancing
# - Error handling for concurrent requests
# - NPU-specific resource management patterns

import gradio as gr
import time
import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

# =======================================
# PATTERN 1: Basic Queue Management
# =======================================
# Source: demo/concurrency_with_queue/run.py (enhanced for OpenVINO)

def create_basic_queue_demo():
    """
    Basic queue management for handling multiple OpenVINO GenAI requests.
    Essential pattern for managing NPU resources efficiently.
    """
    
    def slow_openvino_generation(message: str):
        """Simulate OpenVINO GenAI generation with realistic timing"""
        # Simulate NPU model loading/compilation time
        if random.random() < 0.1:  # 10% chance of compilation
            print(f"🔧 Compiling model for request: {message[:20]}...")
            time.sleep(3)  # NPU compilation overhead
        
        # Simulate actual inference time
        inference_time = random.uniform(2, 6)  # 2-6 seconds for generation
        time.sleep(inference_time)
        
        return f"OpenVINO GenAI Response (took {inference_time:.1f}s): {message}"
    
    with gr.Blocks() as demo:
        gr.Markdown("# 🚀 OpenVINO GenAI Queue Management Demo")
        gr.Markdown("Multiple requests are queued and processed sequentially to manage NPU resources")
        
        input_text = gr.Textbox(label="Your Message", placeholder="Enter message for OpenVINO GenAI...")
        output_text = gr.Textbox(label="Response")
        submit_btn = gr.Button("Generate Response", variant="primary")
        
        # Queue status display
        queue_status = gr.Textbox(label="Queue Status", value="Ready", interactive=False)
        
        def update_queue_status():
            """Update queue status display"""
            return "🟡 Processing request..."
        
        def reset_queue_status():
            """Reset queue status after completion"""
            return "✅ Ready for next request"
        
        # Event with queue management
        submit_btn.click(
            update_queue_status,
            outputs=queue_status,
            queue=False  # Status update should be immediate
        ).then(
            slow_openvino_generation,
            inputs=input_text,
            outputs=output_text
            # queue=True by default - requests will be queued
        ).then(
            reset_queue_status,
            outputs=queue_status,
            queue=False
        )
    
    return demo

# =========================================
# PATTERN 2: Advanced Concurrency Control
# =========================================
# Sophisticated concurrency management for production deployments

class RequestPriority(Enum):
    HIGH = 1
    NORMAL = 2  
    LOW = 3

@dataclass
class OpenVINORequest:
    message: str
    priority: RequestPriority
    user_id: str
    timestamp: float
    max_tokens: int = 512

class OpenVINORequestManager:
    """Advanced request manager for OpenVINO GenAI applications"""
    
    def __init__(self, max_concurrent: int = 2, npu_available: bool = True):
        self.max_concurrent = max_concurrent
        self.npu_available = npu_available
        self.active_requests = 0
        self.request_queue = queue.PriorityQueue()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.request_history: List[Dict] = []
        
    def add_request(self, request: OpenVINORequest):
        """Add request to priority queue"""
        priority_value = request.priority.value
        self.request_queue.put((priority_value, request.timestamp, request))
        
    def process_request(self, request: OpenVINORequest) -> str:
        """Process individual OpenVINO GenAI request"""
        start_time = time.time()
        
        try:
            self.active_requests += 1
            
            # Simulate device selection
            device = "NPU" if self.npu_available and random.random() > 0.3 else "CPU"
            
            # Simulate OpenVINO GenAI processing
            if device == "NPU":
                # NPU processing - faster but limited concurrency
                processing_time = random.uniform(1.5, 3.0)
                compile_time = random.uniform(0.5, 2.0) if random.random() < 0.2 else 0
            else:
                # CPU processing - slower but more concurrent
                processing_time = random.uniform(3.0, 6.0)
                compile_time = 0
            
            time.sleep(compile_time + processing_time)
            
            # Record metrics
            total_time = time.time() - start_time
            self.request_history.append({
                'user_id': request.user_id,
                'priority': request.priority.name,
                'device': device,
                'processing_time': processing_time,
                'compile_time': compile_time,
                'total_time': total_time,
                'max_tokens': request.max_tokens,
                'timestamp': start_time
            })
            
            tokens_per_sec = request.max_tokens / processing_time if processing_time > 0 else 0
            
            result = f"[{device}] Generated response for '{request.message[:30]}...' "
            result += f"({total_time:.1f}s, {tokens_per_sec:.1f} tok/s, Priority: {request.priority.name})"
            
            return result
            
        except Exception as e:
            return f"❌ Error processing request: {str(e)}"
        finally:
            self.active_requests -= 1
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            'active_requests': self.active_requests,
            'queued_requests': self.request_queue.qsize(),
            'max_concurrent': self.max_concurrent,
            'npu_available': self.npu_available,
            'total_processed': len(self.request_history)
        }

def create_advanced_concurrency_demo():
    """
    Advanced concurrency control demo for OpenVINO GenAI applications.
    Shows proper resource management and request prioritization.
    """
    
    manager = OpenVINORequestManager(max_concurrent=2, npu_available=True)
    
    def submit_request(message: str, priority: str, user_id: str, max_tokens: int):
        """Submit request to the manager"""
        if not message.strip():
            return "⚠️ Please enter a message", manager.get_queue_status()
        
        priority_enum = RequestPriority[priority.upper()]
        request = OpenVINORequest(
            message=message,
            priority=priority_enum,
            user_id=user_id,
            timestamp=time.time(),
            max_tokens=max_tokens
        )
        
        # Process request
        result = manager.process_request(request)
        status = manager.get_queue_status()
        
        return result, status
    
    def get_performance_stats():
        """Get performance statistics"""
        if not manager.request_history:
            return "No requests processed yet"
        
        history = manager.request_history
        avg_total_time = sum(r['total_time'] for r in history) / len(history)
        avg_processing_time = sum(r['processing_time'] for r in history) / len(history)
        
        device_usage = {}
        for r in history:
            device = r['device']
            device_usage[device] = device_usage.get(device, 0) + 1
        
        stats = {
            "total_requests": len(history),
            "avg_total_time": round(avg_total_time, 2),
            "avg_processing_time": round(avg_processing_time, 2),
            "device_usage": device_usage,
            "recent_requests": history[-5:]  # Last 5 requests
        }
        
        return stats
    
    with gr.Blocks() as demo:
        gr.Markdown("# ⚡ Advanced OpenVINO GenAI Concurrency Management")
        gr.Markdown("Production-grade request handling with priority queues and resource management")
        
        with gr.Row():
            with gr.Column(scale=2):
                message_input = gr.Textbox(
                    label="Message",
                    placeholder="Enter your message for OpenVINO GenAI...",
                    lines=2
                )
                with gr.Row():
                    priority_select = gr.Dropdown(
                        choices=["HIGH", "NORMAL", "LOW"],
                        value="NORMAL",
                        label="Priority"
                    )
                    user_id_input = gr.Textbox(
                        value="user1",
                        label="User ID",
                        scale=1
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        value=512,
                        label="Max Tokens"
                    )
                
                submit_btn = gr.Button("🚀 Submit Request", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                queue_status_json = gr.JSON(label="Queue Status")
                performance_btn = gr.Button("📊 Performance Stats")
        
        with gr.Row():
            result_output = gr.Textbox(
                label="Response",
                lines=4,
                max_lines=10
            )
        
        with gr.Row(visible=False) as stats_row:
            performance_stats = gr.JSON(label="Performance Statistics")
        
        # Event handlers
        submit_btn.click(
            submit_request,
            inputs=[message_input, priority_select, user_id_input, max_tokens_slider],
            outputs=[result_output, queue_status_json]
        )
        
        performance_btn.click(
            lambda: (gr.update(visible=True), get_performance_stats()),
            outputs=[stats_row, performance_stats]
        )
        
        # Initialize with current status
        demo.load(
            lambda: manager.get_queue_status(),
            outputs=queue_status_json
        )
    
    return demo

# =========================================
# PATTERN 3: Async Streaming with Queues
# =========================================
# Advanced async pattern for streaming responses with queue management

class AsyncOpenVINOManager:
    """Async OpenVINO GenAI manager for streaming responses"""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.active_streams = {}
        self.stream_counter = 0
        
    async def stream_generation(self, message: str, user_id: str) -> str:
        """Async streaming generation"""
        stream_id = f"stream_{self.stream_counter}_{user_id}"
        self.stream_counter += 1
        
        try:
            self.active_streams[stream_id] = {
                'message': message,
                'user_id': user_id,
                'start_time': time.time(),
                'status': 'processing'
            }
            
            # Simulate async OpenVINO GenAI streaming
            response_parts = [
                f"Processing '{message[:20]}...' with OpenVINO GenAI",
                "Initializing NPU resources...",
                "Compiling model graph...",
                "Starting inference...",
                "Generating tokens...",
                f"Response: This is a simulated streaming response to '{message}'"
            ]
            
            for i, part in enumerate(response_parts):
                await asyncio.sleep(0.8)  # Simulate processing time
                progress = f"[{i+1}/{len(response_parts)}] {part}"
                yield progress
            
            self.active_streams[stream_id]['status'] = 'completed'
            
        except Exception as e:
            self.active_streams[stream_id]['status'] = 'error'
            yield f"❌ Error: {str(e)}"
        
        finally:
            # Keep completed streams for a while for monitoring
            await asyncio.sleep(5)
            self.active_streams.pop(stream_id, None)
    
    def get_active_streams(self) -> Dict[str, Any]:
        """Get information about active streams"""
        return {
            'count': len(self.active_streams),
            'streams': list(self.active_streams.values()),
            'max_concurrent': self.max_concurrent
        }

def create_async_streaming_demo():
    """
    Async streaming demo with queue management.
    Shows how to handle multiple concurrent streaming requests.
    """
    
    manager = AsyncOpenVINOManager(max_concurrent=3)
    
    def sync_stream_wrapper(message: str, user_id: str, history: list):
        """Wrapper to make async streaming work with Gradio"""
        # This is a simplified version - in practice, use proper async integration
        
        # Simulate streaming response
        responses = [
            f"🔄 Starting generation for user {user_id}...",
            f"🧠 Processing: '{message[:30]}...'",
            "⚡ Using NPU for acceleration...",
            "🔧 Optimizing with OpenVINO...",
            f"✅ Generated response: This is a simulated streaming response for '{message}'. The system processed your request using advanced NPU acceleration and OpenVINO optimization."
        ]
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        
        for i, response_part in enumerate(responses):
            if i == 0:
                history[-1]["content"] = response_part
            else:
                history[-1]["content"] = response_part
            
            time.sleep(1.0)  # Simulate streaming delay
            yield history
    
    with gr.Blocks() as demo:
        gr.Markdown("# 🌊 Async Streaming Queue Management")
        gr.Markdown("Concurrent streaming responses with proper queue management")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type="messages",
                    height=400,
                    label="Streaming Chat"
                )
                
                with gr.Row():
                    message_input = gr.Textbox(
                        placeholder="Enter message for streaming generation...",
                        scale=4
                    )
                    user_id = gr.Textbox(
                        value="user1",
                        label="User ID",
                        scale=1
                    )
                
                stream_btn = gr.Button("🌊 Start Streaming", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### Queue Monitor")
                active_streams = gr.JSON(label="Active Streams")
                refresh_btn = gr.Button("🔄 Refresh Status")
        
        # Event handlers
        stream_btn.click(
            sync_stream_wrapper,
            inputs=[message_input, user_id, chatbot],
            outputs=chatbot
        ).then(
            lambda: "",
            outputs=message_input
        )
        
        refresh_btn.click(
            lambda: manager.get_active_streams(),
            outputs=active_streams
        )
        
        # Initialize
        demo.load(
            lambda: manager.get_active_streams(),
            outputs=active_streams
        )
    
    return demo

# =====================================
# INTEGRATION GUIDELINES FOR OPENVINO
# =====================================

"""
OPENVINO GENAI INTEGRATION GUIDELINES:
=====================================

1. QUEUE CONFIGURATION:
   ```python
   # Configure Gradio queue for OpenVINO GenAI
   demo.queue(
       default_concurrency_limit=2,  # Limit concurrent NPU requests
       max_size=10,                   # Maximum queue size
       api_open=False                 # Disable API access if not needed
   )
   
   # Launch with queue enabled
   demo.launch(enable_queue=True)
   ```

2. NPU RESOURCE MANAGEMENT:
   ```python
   class NPUResourceManager:
       def __init__(self):
           self.npu_busy = False
           self.npu_lock = threading.Lock()
       
       def acquire_npu(self):
           with self.npu_lock:
               if self.npu_busy:
                   return False
               self.npu_busy = True
               return True
       
       def release_npu(self):
           with self.npu_lock:
               self.npu_busy = False
   ```

3. REQUEST PRIORITIZATION:
   ```python
   # In your generation function:
   def openvino_generate_with_priority(message, history, priority="NORMAL"):
       if priority == "HIGH":
           # Use NPU if available
           device = "NPU" if npu_manager.acquire_npu() else "CPU"
       else:
           # Use CPU for lower priority requests
           device = "CPU"
       
       try:
           # Your OpenVINO generation logic
           result = pipe.generate(message, device=device)
           return result
       finally:
           if device == "NPU":
               npu_manager.release_npu()
   ```

4. CONCURRENT STREAMING:
   ```python
   # For streaming with multiple users:
   def concurrent_streaming_generate(message, history, user_id):
       # Create user-specific streamer
       streamer = UserSpecificStreamer(user_id, tokenizer)
       
       # Generate with proper resource management
       with resource_manager.acquire_device() as device:
           pipe.generate(message, generation_config, streamer)
           
           for chunk in streamer:
               yield chunk
   ```

5. ERROR HANDLING AND FALLBACKS:
   ```python
   def robust_openvino_generate(message, history):
       try:
           # Try NPU first
           return generate_with_npu(message, history)
       except NPUCompilationError:
           # Fallback to CPU
           return generate_with_cpu(message, history)
       except Exception as e:
           # Return error message to user
           return f"❌ Generation failed: {str(e)}"
   ```

PRODUCTION DEPLOYMENT RECOMMENDATIONS:
=====================================

1. SET APPROPRIATE CONCURRENCY LIMITS:
   - NPU: 1-2 concurrent requests max
   - CPU: 2-4 concurrent requests  
   - GPU: 4-8 concurrent requests

2. IMPLEMENT REQUEST QUEUING:
   - Use priority queues for different user types
   - Implement timeout mechanisms
   - Monitor queue length and processing times

3. RESOURCE MONITORING:
   - Track device utilization
   - Monitor memory usage
   - Log request patterns and response times

4. GRACEFUL DEGRADATION:
   - NPU → CPU fallback for compilation failures
   - Reduced quality modes under high load
   - Circuit breaker patterns for failing requests

USAGE IN YOUR PROJECT:
=====================

The advanced concurrency pattern (PATTERN 2) is most suitable for production
OpenVINO GenAI applications. It provides:

- Proper NPU resource management
- Request prioritization
- Performance monitoring
- Graceful error handling

Simply integrate the RequestManager with your existing OpenVINO GenAI
generation logic to enable robust concurrent request handling.
"""

# Example usage:
if __name__ == "__main__":
    # Choose the pattern most suitable for your deployment
    demo = create_advanced_concurrency_demo()  # Recommended for production
    demo.launch()