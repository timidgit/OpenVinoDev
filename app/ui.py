"""
User Interface Module  
====================

Gradio interface creation with advanced features including dynamic system prompts,
RAG document upload, performance monitoring, and professional styling.
"""

import os
import gradio as gr
from typing import Dict, Any

from .config import get_config
from .chat import enhanced_llm_chat, rag_system, RAG_AVAILABLE

# Agent system import
try:
    from .agent import AGENT_AVAILABLE, get_agent
    print("✅ Agent UI components loaded")
except ImportError:
    AGENT_AVAILABLE = False
    get_agent = lambda: None
    print("⚠️ Agent UI components not available")
from .streamer import streaming_metrics

# Import enhanced context patterns
import sys
context_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "context")
sys.path.insert(0, context_path)

# Import Qwen3-specific optimizations
try:
    from qwen3_model_context.npu_optimization import QWEN3_NPU_PROFILES
    from qwen3_model_context.model_architecture import QWEN3_8B_ARCHITECTURE
    from qwen3_model_context.special_tokens import QWEN3_SPECIAL_TOKENS
    ENHANCED_CONTEXT_AVAILABLE = True
except ImportError:
    ENHANCED_CONTEXT_AVAILABLE = False
    QWEN3_NPU_PROFILES = {}
    QWEN3_8B_ARCHITECTURE = {}


# System prompt management
DEFAULT_SYSTEM_PROMPT = """You are a helpful, concise AI assistant powered by Phi-3-mini-128k-instruct running on Intel NPU via OpenVINO GenAI. 

Key behaviors:
- Provide accurate, well-structured responses
- Be concise but comprehensive 
- Use clear formatting when helpful
- Acknowledge when you're uncertain
- Optimize for NPU constraints (prefer shorter, focused responses)

You excel at: reasoning, coding, analysis, creative writing, and technical explanations."""

# Current system prompt (can be modified by user)
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

# Global references (will be set by main.py)
pipe = None
device_used = "NPU"
config_used = "enhanced"
load_time = 0.0


def create_enhanced_interface():
    """Create production-ready Gradio interface with advanced features"""
    
    config = get_config()
    
    # Custom CSS for professional appearance
    custom_css = """
    .gradio-container { max-width: 1400px; margin: auto; }
    .chatbot { height: 650px; }
    .metrics-panel { 
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #dee2e6;
    }
    .system-info {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 6px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    .warning-banner {
        background: #fff3cd;
        padding: 8px;
        border-radius: 4px;
        border-left: 4px solid #ffc107;
        margin: 5px 0;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate"
        ),
        title="Enhanced Phi-3 Chat",
        css=custom_css,
    ) as demo:
        
        # Header with system status
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(f"""
                # 🤖 Enhanced Phi-3 Chat System
                
                **Production-Ready Implementation with Complete Optimization**
                """)
            
            with gr.Column(scale=2, elem_classes=["system-info"]):
                system_status = gr.Markdown(f"""
                **Device**: {device_used} | **Config**: {config_used}  
                **Model**: Phi-3-mini-128k-instruct INT4 | **Load Time**: {load_time:.1f}s  
                **Enhanced Context**: {'✅ Active' if ENHANCED_CONTEXT_AVAILABLE else '⚠️ Fallback'}  
                **Profile**: {config.get('deployment', 'npu_profile', 'balanced').title()}
                """)
        
        # Warning banner if fallback mode
        if not ENHANCED_CONTEXT_AVAILABLE:
            gr.Markdown("""
            <div class="warning-banner">
            ⚠️ <strong>Fallback Mode</strong>: Enhanced context not loaded. Some optimizations may be limited.
            </div>
            """)
        
        # Main chat interface using official ChatInterface pattern
        chatbot = gr.Chatbot(
            label=f"Conversation (Phi-3-mini-128k on {device_used})",
            height=650,
            type='messages',
            avatar_images=(None, "🤖"),
            show_copy_button=True,
            show_share_button=False,
            bubble_full_width=False,
            render_markdown=True
        )
        
        # System prompt control
        with gr.Accordion("🎯 System Prompt Configuration", open=False):
            system_prompt_input = gr.Textbox(
                value=SYSTEM_PROMPT,
                lines=6,
                label="System Prompt",
                placeholder="Configure the AI's behavior and persona...",
                interactive=True,
                info="This prompt sets the AI's behavior, expertise, and response style. Changes take effect after clearing the chat."
            )
            
            with gr.Row():
                reset_prompt_btn = gr.Button("🔄 Reset to Default", size="sm")
                apply_prompt_btn = gr.Button("✅ Apply & Clear Chat", variant="primary", size="sm")
        
        # Document upload for RAG
        with gr.Accordion("📚 Document Upload (RAG)", open=False):
            with gr.Row():
                with gr.Column(scale=3):
                    # Dynamic file types based on parsing capabilities
                    if rag_system.advanced_parsing:
                        supported_types = [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".pdf", ".docx", ".pptx"]
                        upload_label = "Upload Documents (Advanced Parsing Enabled)"
                    else:
                        supported_types = [".txt", ".md", ".py", ".js", ".html", ".css", ".json"]
                        upload_label = "Upload Documents (Basic Text Processing)"
                    
                    file_upload = gr.File(
                        label=upload_label,
                        file_types=supported_types,
                        file_count="multiple",
                        interactive=True
                    )
                    
                with gr.Column(scale=2):
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        interactive=False,
                        placeholder="No documents uploaded"
                    )
            
            with gr.Row():
                clear_docs_btn = gr.Button("🗑️ Clear Documents", variant="secondary", size="sm")
                rag_status_btn = gr.Button("📊 RAG Status", size="sm")
            
            # Dynamic check for RAG availability 
            try:
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                rag_available_now = True
            except ImportError:
                rag_available_now = False
            
            if not rag_available_now:
                gr.Markdown("""
                ⚠️ **RAG not available**: Install dependencies with:
                ```bash
                # Basic RAG functionality
                pip install langchain faiss-cpu sentence-transformers
                
                # Advanced features (Phase 3)
                pip install unstructured[local-inference] langchain-unstructured torch
                ```
                """)

        # Agent Tools Section (Phase 3.3)
        with gr.Accordion("🤖 AI Agent Tools", open=False):
            if AGENT_AVAILABLE:
                agent_tools_display = gr.Markdown("""
                **Available Agent Tools:**
                
                🧮 **Calculator**: Perform mathematical calculations  
                📅 **Date/Time**: Get current time, dates, and calculate date differences  
                🔍 **Web Search**: Search for information (mock implementation)  
                📊 **Text Analysis**: Analyze text for word count, readability, etc.
                
                **Usage**: Simply ask naturally! The agent will automatically use tools when needed.
                
                **Examples:**
                - "What's 25 * 4 + 10?"
                - "What time is it?"
                - "Analyze this text: [your text]"
                - "Search for information about AI"
                """)
                
                with gr.Row():
                    agent_status_btn = gr.Button("🤖 Agent Status", variant="secondary", size="sm")
                    agent_tools_btn = gr.Button("🛠️ List Tools", variant="secondary", size="sm")
                
                agent_status_output = gr.Textbox(
                    label="Agent System Status",
                    interactive=False,
                    visible=False,
                    max_lines=5
                )
            else:
                gr.Markdown("""
                ⚠️ **Agent system not available**: Install dependencies with:
                ```bash
                pip install langchain-core langchain-experimental requests python-dateutil
                ```
                
                Once installed, the AI will be able to use tools like calculators, web search, 
                and text analysis automatically based on your requests.
                """)

        # Advanced Generation Settings
        with gr.Accordion("⚙️ Advanced Generation Settings", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        step=0.1,
                        value=config.get('generation', 'temperature', 0.6),
                        label="Temperature",
                        info="Controls randomness (0.1=focused, 2.0=creative)"
                    )
                    
                with gr.Column(scale=1):
                    top_p_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=config.get('generation', 'top_p', 0.95),
                        label="Top-p (Nucleus Sampling)",
                        info="Cumulative probability threshold"
                    )
                
                with gr.Column(scale=1):
                    max_tokens_number = gr.Number(
                        minimum=50,
                        maximum=2048,
                        step=50,
                        value=config.get('generation', 'max_new_tokens', 512),
                        label="Max New Tokens",
                        info="Maximum tokens to generate"
                    )
            
            # Generation settings controls
            with gr.Row():
                reset_gen_settings_btn = gr.Button("🔄 Reset to Defaults", variant="secondary", size="sm")
                apply_gen_settings_btn = gr.Button("✅ Apply Settings", variant="primary", size="sm")
            
            # Current generation settings display
            gen_settings_status = gr.Textbox(
                value=f"Temperature: {config.get('generation', 'temperature', 0.6)}, Top-p: {config.get('generation', 'top_p', 0.95)}, Max tokens: {config.get('generation', 'max_new_tokens', 512)}",
                label="Current Settings",
                interactive=False,
                max_lines=1
            )

        # Input controls
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder=f"Chat with Phi-3 on {device_used} (max {config.get('ui', 'max_message_length', 2000)} chars)...",
                scale=7,
                max_lines=4,
                show_label=False,
                container=False
            )
            
            with gr.Column(scale=1):
                send_btn = gr.Button("💬 Send", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
        
        # Advanced controls panel
        with gr.Row():
            with gr.Column(scale=2):
                metrics_btn = gr.Button("📊 Performance Metrics", variant="secondary")
                system_btn = gr.Button("ℹ️ System Info", variant="secondary")
            
            with gr.Column(scale=2):
                if ENHANCED_CONTEXT_AVAILABLE and QWEN3_NPU_PROFILES:
                    profile_selector = gr.Dropdown(
                        choices=list(QWEN3_NPU_PROFILES.keys()),
                        value=config.get('deployment', 'npu_profile', 'balanced'),
                        label="NPU Profile",
                        interactive=False  # Would need restart to change
                    )
                
                reset_metrics_btn = gr.Button("🔄 Reset Metrics", variant="secondary")
        
        # Collapsible metrics panel
        with gr.Row(visible=False) as metrics_row:
            with gr.Column(elem_classes=["metrics-panel"]):
                gr.Markdown("### 📊 Real-time Performance Metrics")
                metrics_json = gr.JSON(label="System Metrics", container=True)
                
                if ENHANCED_CONTEXT_AVAILABLE:
                    gr.Markdown("### 🎯 Model-Specific Stats")
                    phi3_stats = gr.JSON(label="Token Filtering & Processing", container=True)
        
        # Examples section
        with gr.Row():
            gr.Examples(
                examples=[
                    "Explain quantum computing in simple terms",
                    "Write a Python function to implement quicksort", 
                    "What are the advantages of using Intel NPU for AI inference?",
                    "Compare different neural network architectures",
                    "Help me debug this code: def factorial(n): return n * factorial(n)",
                    "Explain the concept of attention in transformer models",
                    "What does the uploaded document say about...?",
                    "Summarize the key points from the uploaded files",
                    # Agent examples
                    "What's 15 * 23 + 47?",
                    "What time is it right now?", 
                    "Calculate the square root of 144",
                    "Analyze this text: The quick brown fox jumps over the lazy dog",
                    "What's tomorrow's date?"
                ] if AGENT_AVAILABLE else [
                    "Explain quantum computing in simple terms",
                    "Write a Python function to implement quicksort",
                    "What are the advantages of using Intel NPU for AI inference?",
                    "Compare different neural network architectures", 
                    "Help me debug this code: def factorial(n): return n * factorial(n)",
                    "Explain the concept of attention in transformer models",
                    "What does the uploaded document say about...?",
                    "Summarize the key points from the uploaded files"
                ],
                inputs=msg_input,
                label="💡 Example Questions (Upload documents for context-aware answers)"
            )
        
        # Global generation settings storage
        generation_settings = {
            'temperature': config.get('generation', 'temperature', 0.6),
            'top_p': config.get('generation', 'top_p', 0.95),
            'max_new_tokens': config.get('generation', 'max_new_tokens', 512)
        }
        
        # Event handlers with enhanced functionality
        def handle_send(message, history):
            """Handle send with proper session management"""
            return enhanced_llm_chat(message, history, generation_settings)
        
        def handle_clear(current_system_prompt):
            """Handle clear with proper session reset"""
            global SYSTEM_PROMPT
            try:
                # Update global system prompt if changed
                if current_system_prompt.strip():
                    SYSTEM_PROMPT = current_system_prompt.strip()
                else:
                    SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
                
                # End current session and start new one
                pipe.finish_chat()
                pipe.start_chat(SYSTEM_PROMPT)
                print("🔄 Chat session reset with updated system prompt")
                return [], "", SYSTEM_PROMPT
            except Exception as e:
                print(f"⚠️ Session reset error: {e}")
                return [], "", current_system_prompt
        
        def show_metrics():
            """Display comprehensive performance metrics"""
            base_metrics = streaming_metrics.get_summary()
            
            qwen3_specific = {}
            if ENHANCED_CONTEXT_AVAILABLE:
                qwen3_specific = {
                    "Enhanced Features": "Active",
                    "NPUW Profile": config.get('deployment', 'npu_profile', 'balanced'),
                    "Model Architecture": f"Phi-3-mini-128k-instruct",
                    "Max Context": f"{QWEN3_8B_ARCHITECTURE.get('max_position_embeddings', 40960):,} tokens",
                    "Special Tokens Available": len(QWEN3_SPECIAL_TOKENS) if 'QWEN3_SPECIAL_TOKENS' in globals() else 0
                }
            else:
                qwen3_specific = {
                    "Enhanced Features": "Fallback Mode",
                    "Note": "Install enhanced context for full optimization"
                }
            
            return (
                gr.update(value=base_metrics, visible=True),
                gr.update(value=qwen3_specific, visible=True) if ENHANCED_CONTEXT_AVAILABLE else gr.update(visible=False),
                gr.update(visible=True)
            )
        
        def show_system_info():
            """Display comprehensive system information"""
            config = get_config()
            model_path = config.get("model", "path")
            cache_dir = config.get("deployment", "cache_directory")
            npu_profile = config.get("deployment", "npu_profile", "balanced")
            
            info_text = f"""
            ## 🖥️ System Configuration
            
            **Hardware & Device:**
            - Target Device: {device_used}
            - Configuration: {config_used}
            - Cache Directory: `{cache_dir}`
            - NPU Profile: {npu_profile}
            
            **Model Details:**
            - Model: Phi-3-mini-128k-instruct INT4 Quantized
            - Path: `{model_path}`
            - Load Time: {load_time:.1f} seconds
            - Tokenizer: HuggingFace AutoTokenizer
            
            **OpenVINO GenAI Configuration:**
            - API Mode: Stateful (start_chat/finish_chat)
            - Conversation Management: Automatic KV-cache
            - Token Limits: {config.get('ui', 'max_conversation_tokens', 1800)} (conversation), {config.get('ui', 'max_message_length', 400)} (message)
            - Generation: Temperature={config.get('generation', 'temperature', 0.6)}, Top-p={config.get('generation', 'top_p', 0.95)}
            
            **Enhanced Features:**
            {"✅ Complete Phi-3 NPUW optimization" if ENHANCED_CONTEXT_AVAILABLE else "⚠️ Basic NPUW configuration"}
            {"✅ Advanced special token filtering" if ENHANCED_CONTEXT_AVAILABLE else "⚠️ Basic token filtering"}
            {"✅ Phi-3-specific optimizations" if ENHANCED_CONTEXT_AVAILABLE else "⚠️ Standard templates"}
            {"✅ Advanced performance monitoring" if ENHANCED_CONTEXT_AVAILABLE else "⚠️ Basic metrics"}
            {"✅ RAG document processing" if rag_system.available else "⚠️ RAG not available"}
            {"✅ AI Agent with function-calling" if AGENT_AVAILABLE else "⚠️ Agent system not available"}
            
            **Performance Targets (NPU):**
            - Load Time: <90s (first run), <30s (cached)
            - First Token: <2s latency
            - Generation: 15-25 tokens/second
            - Memory: Optimized for NPU constraints
            """
            
            gr.Info(info_text)
        
        def reset_metrics():
            """Reset performance metrics"""
            streaming_metrics.reset()
            gr.Info("📊 Performance metrics reset successfully")
        
        def reset_system_prompt():
            """Reset system prompt to default"""
            return DEFAULT_SYSTEM_PROMPT
        
        def apply_system_prompt(new_prompt):
            """Apply new system prompt and clear chat"""
            global SYSTEM_PROMPT
            if new_prompt.strip():
                SYSTEM_PROMPT = new_prompt.strip()
            else:
                SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
            
            try:
                pipe.finish_chat()
                pipe.start_chat(SYSTEM_PROMPT)
                gr.Info("✅ System prompt updated and chat cleared")
                return [], "", SYSTEM_PROMPT
            except Exception as e:
                gr.Warning(f"⚠️ Error applying prompt: {e}")
                return [], "", new_prompt
        
        def handle_file_upload(files):
            """Handle uploaded files for RAG processing"""
            if not files:
                return "No files selected"
            
            results = []
            for file in files:
                if file is None:
                    continue
                
                file_name = os.path.basename(file.name)
                result = rag_system.process_uploaded_file(file.name, file_name)
                results.append(result)
            
            return "\n\n".join(results)
        
        def clear_documents():
            """Clear all uploaded documents"""
            result = rag_system.clear_documents()
            gr.Info(result)
            return result
        
        def show_rag_status():
            """Show RAG system status"""
            status = rag_system.get_status()
            gr.Info(f"RAG Status: {status}")
            return str(status)
        
        def reset_generation_settings():
            """Reset generation settings to defaults"""
            default_temp = config.get('generation', 'temperature', 0.6)
            default_top_p = config.get('generation', 'top_p', 0.95)
            default_max_tokens = config.get('generation', 'max_new_tokens', 512)
            
            generation_settings.update({
                'temperature': default_temp,
                'top_p': default_top_p,
                'max_new_tokens': default_max_tokens
            })
            
            status_text = f"Temperature: {default_temp}, Top-p: {default_top_p}, Max tokens: {default_max_tokens}"
            gr.Info("🔄 Generation settings reset to defaults")
            return default_temp, default_top_p, default_max_tokens, status_text
        
        def apply_generation_settings(temp, top_p, max_tokens):
            """Apply new generation settings"""
            generation_settings.update({
                'temperature': temp,
                'top_p': top_p,
                'max_new_tokens': int(max_tokens)
            })
            
            status_text = f"Temperature: {temp}, Top-p: {top_p}, Max tokens: {int(max_tokens)}"
            gr.Info("✅ Generation settings applied")
            return status_text
        
        def show_agent_status():
            """Show agent system status and capabilities"""
            if not AGENT_AVAILABLE:
                status = "❌ Agent system not available. Install dependencies to enable."
                gr.Info(status)
                return status, gr.update(visible=True)
            
            agent = get_agent()
            if agent:
                status = f"""✅ Agent system active
                
Available Tools: {len(agent.tools)}
Framework: ReAct (Reasoning + Acting)
Integration: OpenVINO GenAI + LangChain

The agent automatically detects when to use tools based on your questions.
Ask naturally and it will use the appropriate tools to help you!"""
            else:
                status = "⚠️ Agent system loaded but not initialized"
            
            gr.Info("Agent status displayed")
            return status, gr.update(visible=True)
        
        def list_agent_tools():
            """List all available agent tools with descriptions"""
            if not AGENT_AVAILABLE:
                status = "❌ No tools available - agent system not loaded"
                gr.Info(status)
                return status, gr.update(visible=True)
            
            agent = get_agent()
            if agent:
                tools_info = []
                for name, info in agent.tools.items():
                    tools_info.append(f"🔧 **{name}**: {info['description']}")
                
                status = "Available Agent Tools:\n\n" + "\n\n".join(tools_info)
            else:
                status = "⚠️ Agent not initialized"
            
            gr.Info("Tools list displayed")
            return status, gr.update(visible=True)
        
        # Wire up event handlers
        msg_input.submit(handle_send, [msg_input, chatbot], chatbot).then(
            lambda: gr.update(value=""), None, [msg_input]
        )
        
        send_btn.click(handle_send, [msg_input, chatbot], chatbot).then(
            lambda: gr.update(value=""), None, [msg_input]
        )
        
        clear_btn.click(handle_clear, [system_prompt_input], [chatbot, msg_input, system_prompt_input])
        
        metrics_btn.click(
            show_metrics, 
            None, 
            [metrics_json, phi3_stats if ENHANCED_CONTEXT_AVAILABLE else None, metrics_row]
        )
        
        system_btn.click(show_system_info, None, None)
        reset_metrics_btn.click(reset_metrics, None, None)
        
        # System prompt event handlers
        reset_prompt_btn.click(reset_system_prompt, None, [system_prompt_input])
        apply_prompt_btn.click(
            apply_system_prompt, 
            [system_prompt_input], 
            [chatbot, msg_input, system_prompt_input]
        )
        
        # RAG event handlers - always enable, will show error if RAG not available
        file_upload.upload(handle_file_upload, [file_upload], [upload_status])
        clear_docs_btn.click(clear_documents, None, [upload_status])
        rag_status_btn.click(show_rag_status, None, [upload_status])
        
        # Generation settings event handlers
        reset_gen_settings_btn.click(
            reset_generation_settings,
            None,
            [temperature_slider, top_p_slider, max_tokens_number, gen_settings_status]
        )
        
        apply_gen_settings_btn.click(
            apply_generation_settings,
            [temperature_slider, top_p_slider, max_tokens_number],
            [gen_settings_status]
        )
        
        # Agent event handlers (if available)
        if AGENT_AVAILABLE:
            agent_status_btn.click(
                show_agent_status,
                None,
                [agent_status_output, agent_status_output]
            )
            
            agent_tools_btn.click(
                list_agent_tools,
                None,
                [agent_status_output, agent_status_output]
            )
        
        # Initialize chat session when interface loads
        def initialize_session():
            """Initialize chat session with system prompt"""
            try:
                pipe.start_chat(SYSTEM_PROMPT)
                print("✅ Chat session initialized with system prompt")
            except Exception as e:
                print(f"⚠️ Session initialization error: {e}")
        
        demo.load(initialize_session, None, None)
    
    return demo


def initialize_ui_globals(pipeline, device, config, load_time_val):
    """Initialize global UI variables"""
    global pipe, device_used, config_used, load_time
    pipe = pipeline
    device_used = device
    config_used = config
    load_time = load_time_val