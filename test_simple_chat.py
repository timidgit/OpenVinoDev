#!/usr/bin/env python3
"""
Ultra-simple Gradio chat test to isolate the format issue
"""

import gradio as gr

def simple_chat_test(message, history):
    """Minimal streaming test that yields correct format"""
    print(f"Input - message: {message}")
    print(f"Input - history: {history}")
    
    # Add new user message to history
    if not isinstance(history, list):
        history = []
    
    # Create new history with user message and empty assistant response
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""}
    ]
    
    # Stream response word by word
    words = ["This", "is", "a", "test", "response"]
    for word in words:
        new_history[-1]["content"] += word + " "
        print(f"Yielding: {new_history}")
        yield new_history

# Simple interface
with gr.Blocks() as demo:
    gr.Markdown("# Simple Chat Test")
    
    chatbot = gr.Chatbot(type='messages')
    msg = gr.Textbox(placeholder="Type a message...")
    
    msg.submit(simple_chat_test, [msg, chatbot], chatbot).then(
        lambda: gr.update(value=""), None, [msg]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7863, debug=True)