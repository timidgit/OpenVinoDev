#!/usr/bin/env python3
"""
Absolute minimal test to isolate Gradio format issue
"""

import gradio as gr

def minimal_test(message, history):
    """Minimal test with hardcoded correct format"""
    print(f"🧪 Input: message='{message}', history={history}")
    
    # Return the absolute minimal correct format
    result = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "response"}
    ]
    
    print(f"🧪 Returning: {repr(result)}")
    return result

def generator_test(message, history):
    """Test with generator (like our main app)"""
    print(f"🧪 Generator Input: message='{message}', history={history}")
    
    def gen():
        result = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"}
        ]
        print(f"🧪 Generator Yielding: {repr(result)}")
        yield result
    
    return gen()

# Test both approaches
with gr.Blocks() as demo:
    with gr.Tab("Direct Return"):
        chatbot1 = gr.Chatbot(type='messages', height=300)
        msg1 = gr.Textbox(placeholder="Test direct return...")
        msg1.submit(minimal_test, [msg1, chatbot1], chatbot1).then(
            lambda: gr.update(value=""), None, [msg1]
        )
    
    with gr.Tab("Generator Return"):
        chatbot2 = gr.Chatbot(type='messages', height=300)
        msg2 = gr.Textbox(placeholder="Test generator return...")
        msg2.submit(generator_test, [msg2, chatbot2], chatbot2).then(
            lambda: gr.update(value=""), None, [msg2]
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862)