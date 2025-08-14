#!/usr/bin/env python3
"""
Quick test to verify Gradio chat format compatibility
"""

import gradio as gr

def simple_chat_test(message, history):
    """Simple test function that ensures correct format"""
    print(f"🧪 TEST - Message: {message}")
    print(f"🧪 TEST - History type: {type(history)}")
    print(f"🧪 TEST - History content: {history}")
    
    # Ensure history is correct format
    if not isinstance(history, list):
        history = []
    
    # Add user message and bot response
    new_history = history.copy()
    new_history.append({"role": "user", "content": message})
    new_history.append({"role": "assistant", "content": f"Echo: {message}"})
    
    print(f"🧪 TEST - Returning: {new_history}")
    return new_history

# Create test interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        type='messages',
        height=400
    )
    msg_input = gr.Textbox(
        placeholder="Test message...",
        show_label=False
    )
    
    msg_input.submit(
        simple_chat_test, 
        [msg_input, chatbot], 
        chatbot
    ).then(
        lambda: gr.update(value=""), 
        None, 
        [msg_input]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)