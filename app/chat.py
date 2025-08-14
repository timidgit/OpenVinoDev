"""
Chat Processing Module
=====================

Core chat processing logic with RAG integration, input validation,
security features, and comprehensive error handling.
"""

import time
import threading
import re
from typing import Iterator, List, Dict, Tuple, Any
from typing_extensions import TypedDict

import openvino_genai as ov_genai

from .config import get_config
from .streamer import EnhancedLLMStreamer, streaming_metrics

# Agent system import (Phase 3.3)
try:
    from .agent import get_agent, AGENT_AVAILABLE
    print("✅ Agent system loaded")
except ImportError as e:
    print(f"⚠️ Agent system not available: {e}")
    AGENT_AVAILABLE = False
    get_agent = lambda: None

# Type definitions for Gradio ChatInterface compatibility
# ChatHistory is a list of message dictionaries with role and content
ChatHistory = List[Dict[str, str]]

# RAG system imports with fallback
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    try:
        # Try modern import first
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        # Fallback to legacy import
        from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    
    # Advanced document parsing (Phase 3.1)
    try:
        from langchain_unstructured import UnstructuredLoader
        from unstructured.partition.auto import partition
        ADVANCED_PARSING_AVAILABLE = True
        print("✅ Advanced document parsing (unstructured) loaded successfully")
    except ImportError:
        ADVANCED_PARSING_AVAILABLE = False
        print("📝 Advanced parsing not available - install with: pip install unstructured[local-inference] langchain-unstructured")
    
    # Cross-encoder reranking (Phase 3.2)
    try:
        from sentence_transformers import CrossEncoder
        RERANKING_AVAILABLE = True
        print("✅ Cross-encoder reranking loaded successfully")
    except ImportError:
        RERANKING_AVAILABLE = False
        print("📝 Reranking not available - already have sentence-transformers but need torch")
    
    RAG_AVAILABLE = True
    print("✅ RAG dependencies loaded successfully")
except ImportError as e:
    print(f"⚠️ RAG dependencies not available: {e}")
    print("📝 Install with: pip install langchain faiss-cpu sentence-transformers")
    RAG_AVAILABLE = False
    ADVANCED_PARSING_AVAILABLE = False
    RERANKING_AVAILABLE = False


class DocumentRAGSystem:
    """Retrieval-Augmented Generation system for document processing"""
    
    def __init__(self):
        """Initialize RAG system with fallback handling"""
        self.vector_store = None
        self.embeddings = None
        self.text_splitter = None
        self.processed_docs_count = 0
        self.available = RAG_AVAILABLE
        
        # Advanced features
        self.advanced_parsing = ADVANCED_PARSING_AVAILABLE
        self.reranking = RERANKING_AVAILABLE
        self.cross_encoder = None
        
        if RAG_AVAILABLE:
            try:
                # Initialize embeddings model (lightweight for fast loading)
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                
                # Initialize text splitter with optimized settings
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,  # Smaller chunks for better retrieval
                    chunk_overlap=100,  # Overlap for context preservation
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                # Initialize cross-encoder for reranking (Phase 3.2)
                if RERANKING_AVAILABLE:
                    try:
                        self.cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
                        print("✅ Cross-encoder reranker initialized")
                    except Exception as e:
                        print(f"⚠️ Cross-encoder initialization failed: {e}")
                        self.reranking = False
                
                print("✅ RAG system initialized successfully")
                print(f"📊 Features: Advanced parsing={self.advanced_parsing}, Reranking={self.reranking}")
                
            except Exception as e:
                print(f"⚠️ RAG initialization failed: {e}")
                self.available = False
        else:
            print("📝 RAG system not available - install dependencies to enable")
    
    def process_uploaded_file(self, file_path: str, file_name: str) -> str:
        """
        Process uploaded file for RAG retrieval.
        
        Args:
            file_path: Path to the uploaded file
            file_name: Original name of the file
            
        Returns:
            Status message about processing result
        """
        if not self.available:
            return "❌ RAG system not available. Install langchain and faiss-cpu to enable document processing."
        
        try:
            # Use advanced parsing if available (Phase 3.1)
            if self.advanced_parsing and file_name.lower().endswith(('.pdf', '.docx', '.pptx', '.html')):
                try:
                    # Use unstructured for advanced document parsing
                    elements = partition(filename=file_path, strategy="hi_res")
                    text = "\n\n".join([str(element) for element in elements])
                    parsing_method = "Advanced (unstructured)"
                    print(f"📚 Using advanced parsing for {file_name}")
                except Exception as e:
                    print(f"⚠️ Advanced parsing failed for {file_name}, falling back to basic: {e}")
                    # Fallback to basic parsing
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    except UnicodeDecodeError:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            text = f.read()
                    parsing_method = "Basic (fallback)"
            else:
                # Basic text parsing for supported formats
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    # Try with different encoding
                    with open(file_path, 'r', encoding='latin-1') as f:
                        text = f.read()
                parsing_method = "Basic (text)"
            
            if not text.strip():
                return f"⚠️ File '{file_name}' appears to be empty."
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                return f"⚠️ No processable content found in '{file_name}'."
            
            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_texts(
                    texts=chunks, 
                    embedding=self.embeddings,
                    metadatas=[{"source": file_name, "chunk": i} for i in range(len(chunks))]
                )
            else:
                # Add new documents to existing store
                new_store = FAISS.from_texts(
                    texts=chunks, 
                    embedding=self.embeddings,
                    metadatas=[{"source": file_name, "chunk": i} for i in range(len(chunks))]
                )
                self.vector_store.merge_from(new_store)
            
            self.processed_docs_count += 1
            
            return f"✅ Successfully processed '{file_name}' using {parsing_method}: {len(chunks)} chunks created from {len(text):,} characters. Ready to answer questions about this document."
            
        except Exception as e:
            return f"❌ Error processing '{file_name}': {str(e)}"
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """
        Retrieve relevant context for a query with optional cross-encoder reranking.
        
        Args:
            query: User question to find relevant context for
            k: Number of top chunks to retrieve
            
        Returns:
            Concatenated context from relevant document chunks
        """
        if not self.available or self.vector_store is None:
            return ""
        
        try:
            # Use two-stage retrieval with reranking if available (Phase 3.2)
            if self.reranking and self.cross_encoder is not None:
                # Stage 1: Retrieve larger candidate set (e.g., top 20)
                candidate_docs = self.vector_store.similarity_search(query, k=min(20, k*6))
                
                if not candidate_docs:
                    return ""
                
                # Stage 2: Rerank with cross-encoder
                try:
                    query_doc_pairs = [(query, doc.page_content) for doc in candidate_docs]
                    scores = self.cross_encoder.predict(query_doc_pairs)
                    
                    # Sort by reranking scores and take top k
                    scored_docs = list(zip(candidate_docs, scores))
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    docs = [doc for doc, score in scored_docs[:k]]
                    
                    print(f"🔄 Reranked {len(candidate_docs)} candidates → {k} results")
                    
                except Exception as e:
                    print(f"⚠️ Reranking failed, using vector search results: {e}")
                    docs = candidate_docs[:k]
            else:
                # Standard single-stage retrieval
                docs = self.vector_store.similarity_search(query, k=k)
            
            if not docs:
                return ""
            
            # Format context with source attribution
            context_parts = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                content = doc.page_content.strip()
                context_parts.append(f"[From {source}]\n{content}")
            
            return "\n\n---\n\n".join(context_parts)
            
        except Exception as e:
            print(f"⚠️ Context retrieval error: {e}")
            return ""
    
    def clear_documents(self) -> str:
        """Clear all processed documents"""
        self.vector_store = None
        self.processed_docs_count = 0
        return "✅ All documents cleared from memory."
    
    def get_status(self) -> dict:
        """Get current RAG system status with advanced features"""
        return {
            "Available": self.available,
            "Documents Processed": self.processed_docs_count,
            "Vector Store": "Loaded" if self.vector_store is not None else "Empty",
            "Embedding Model": "all-MiniLM-L6-v2" if self.available else "None",
            "Advanced Parsing": "✅ unstructured" if self.advanced_parsing else "❌ basic text only",
            "Cross-Encoder Reranking": "✅ BAAI/bge-reranker-base" if self.reranking else "❌ vector search only",
            "Supported Formats": "PDF, DOCX, PPTX, HTML, TXT, MD, PY, JS, CSS, JSON" if self.advanced_parsing else "TXT, MD, PY, JS, CSS, JSON"
        }


class InputValidator:
    """Security-focused input validation and sanitization"""
    
    @staticmethod
    def validate_message(message: str) -> Tuple[bool, str]:
        """
        Validate user message for security and content policy compliance.
        
        Args:
            message: User input to validate
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if not message or not isinstance(message, str):
            return False, "Empty or invalid message"
        
        # Check for excessively long messages (security)
        if len(message) > 10000:  # Much higher than UI limit
            return False, "Message exceeds maximum length"
        
        # Check for potential injection patterns
        suspicious_patterns = [
            r'<script[^>]*>',  # Script injection
            r'javascript:',     # JavaScript URLs
            r'data:.*base64',   # Data URLs
            r'eval\s*\(',      # Eval calls
            r'exec\s*\(',      # Exec calls
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return False, "Message contains potentially unsafe content"
        
        # Check for excessive special characters (potential encoding attacks)
        special_char_ratio = len([c for c in message if not c.isalnum() and not c.isspace()]) / len(message)
        if special_char_ratio > 0.5:  # More than 50% special characters
            return False, "Message contains excessive special characters"
        
        return True, ""
    
    @staticmethod
    def sanitize_message(message: str) -> str:
        """
        Sanitize user message while preserving readability.
        
        Args:
            message: Raw user input
            
        Returns:
            Sanitized message safe for processing
        """
        # Remove null bytes and control characters (except newlines and tabs)
        sanitized = ''.join(char for char in message if ord(char) >= 32 or char in '\n\t')
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Limit consecutive repeated characters (potential DoS protection)
        sanitized = re.sub(r'(.)\1{10,}', r'\1\1\1', sanitized)
        
        return sanitized.strip()


# Global RAG system instance
rag_system = DocumentRAGSystem()

# Global instances (will be set by main.py)
pipe = None
tokenizer = None


def normalize_history(history) -> ChatHistory:
    """Ensure history is in the correct Gradio messages format."""
    normalized = []
    for entry in history or []:
        if isinstance(entry, dict) and "role" in entry and "content" in entry:
            normalized.append({"role": str(entry["role"]), "content": str(entry["content"])})
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            normalized.append({"role": "user", "content": str(entry[0])})
            normalized.append({"role": "assistant", "content": str(entry[1])})
    return normalized


def create_phi3_generation_config(generation_params: Dict[str, Any] = None) -> ov_genai.GenerationConfig:
    """
    Create optimized generation configuration for Phi-3 with dynamic parameters.
    
    Args:
        generation_params: Optional dict with temperature, top_p, max_new_tokens
    
    Returns:
        Configured GenerationConfig with security-conscious defaults
    """
    gen_config = ov_genai.GenerationConfig()
    config = get_config()
    
    # Load generation settings from configuration (defaults)
    gen_settings = config.get_section("generation")
    
    # Use dynamic parameters if provided, otherwise fall back to config defaults
    if generation_params:
        temperature = generation_params.get('temperature', gen_settings.get("temperature", 0.6))
        top_p = generation_params.get('top_p', gen_settings.get("top_p", 0.95))
        max_new_tokens = generation_params.get('max_new_tokens', gen_settings.get("max_new_tokens", 1024))
    else:
        temperature = gen_settings.get("temperature", 0.6)
        top_p = gen_settings.get("top_p", 0.95)
        max_new_tokens = gen_settings.get("max_new_tokens", 1024)
    
    gen_config.do_sample = gen_settings.get("do_sample", True)
    gen_config.temperature = min(float(temperature), 2.0)  # Security: cap temperature
    gen_config.top_p = min(float(top_p), 1.0)  # Security: cap top_p
    gen_config.top_k = min(gen_settings.get("top_k", 20), 100)  # Security: reasonable top_k
    gen_config.max_new_tokens = min(int(max_new_tokens), 2048)  # Security: limit tokens
    gen_config.repetition_penalty = max(1.0, min(gen_settings.get("repetition_penalty", 1.1), 2.0))  # Security: reasonable range
    
    return gen_config


def process_user_message(message: str, history: ChatHistory) -> Tuple[str, bool]:
    """
    Process user message with smart truncation handling.
    
    Args:
        message: Raw user input message
        history: Current chat conversation history
        
    Returns:
        Tuple of (processed_message, was_truncated)
    """
    config = get_config()
    max_message_length = config.get("ui", "max_message_length", 400)
    original_length = len(message)
    
    # Handle overly long messages
    if original_length > max_message_length:
        # Smart truncation
        if '.' in message:
            sentences = message.split('.')
            truncated = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) + 1 <= max_message_length * 0.85:
                    truncated.append(sentence)
                    current_length += len(sentence) + 1
                else:
                    break
            
            if truncated:
                processed = '. '.join(truncated) + '.'
                if len(processed) < original_length * 0.5:
                    processed = message[:max_message_length-50] + "..."
            else:
                processed = message[:max_message_length-50] + "..."
        else:
            processed = message[:max_message_length-50] + "..."
        
        print(f"📏 Message truncated: {original_length} → {len(processed)} chars")
        return processed, True
    
    return message, False


def prepare_chat_input(message: str, history: ChatHistory) -> Tuple[str, bool, ChatHistory]:
    """
    Prepare and validate chat input with smart message handling and security validation.
    This version includes the definitive fix for the Gradio data format error.
    """
    # Input validation
    if not message.strip():
        # Pass the original history back if the message is empty
        return message, False, history
    
    # Security validation
    is_valid, reason = InputValidator.validate_message(message)
    if not is_valid:
        error_history = [
            {"role": str(item.get("role")), "content": str(item.get("content"))} for item in history
        ]
        error_history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"🚫 Message rejected: {reason}. Please try a different message."}
        ])
        # Note: We are raising an error, but if this were to yield, error_history is now safe
        raise ValueError(f"Security validation failed: {reason}")
    
    # Sanitize and process the user message
    sanitized_message = InputValidator.sanitize_message(message)
    processed_message, was_truncated = process_user_message(sanitized_message, history)
    
    # --- START OF THE DEFINITIVE FIX ---
    # Rebuild the history from scratch on every turn to guarantee a clean state.
    # This prevents any possibility of format corruption from previous turns.
    updated_history = [
        {"role": str(item.get("role")), "content": str(item.get("content"))}
        for item in history if isinstance(item, dict) and "role" in item and "content" in item
    ]
    # --- END OF THE DEFINITIVE FIX ---
    
    if was_truncated:
        # Add truncation warning as a separate exchange
        updated_history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"⚠️ Your message was truncated from {len(message):,} to {len(processed_message)} characters due to NPU memory limits. Processing the truncated version..."}
        ])
    
    # Add current user message with empty bot response placeholder
    updated_history.extend([
        {"role": "user", "content": processed_message},
        {"role": "assistant", "content": ""}
    ])
    
    return processed_message, was_truncated, updated_history


def execute_generation(processed_message: str, streamer: EnhancedLLMStreamer, generation_params: Dict[str, Any] = None) -> bool:
    """
    Execute model generation in a controlled manner.
    
    Args:
        processed_message: Message to generate response for
        streamer: Configured streamer for token processing
        generation_params: Optional dict with temperature, top_p, max_new_tokens
        
    Returns:
        True if generation succeeded, False otherwise
    """
    try:
        generation_config = create_phi3_generation_config(generation_params)
        pipe.generate(processed_message, generation_config, streamer)
        return True
    except Exception as e:
        print(f"❌ Generation error: {e}")
        # Send error through streamer
        error_msg = f"❌ Generation error: {str(e)[:100]}..."
        streamer.text_queue.put(error_msg)
        streamer.text_queue.put(None)
        return False


def stream_response_to_history(streamer: EnhancedLLMStreamer, history: ChatHistory) -> Iterator[ChatHistory]:
    """
    Stream model response tokens to chat history with bulletproof format validation.
    Creates a clean copy on every yield to ensure Gradio compatibility.
    """
    def create_safe_history_copy(hist):
        """Create a guaranteed-safe copy of history for Gradio"""
        safe_history = []
        for entry in hist or []:
            if isinstance(entry, dict) and "role" in entry and "content" in entry:
                safe_history.append({
                    "role": str(entry["role"]),
                    "content": str(entry["content"])
                })
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                safe_history.extend([
                    {"role": "user", "content": str(entry[0])},
                    {"role": "assistant", "content": str(entry[1])}
                ])
        return safe_history
    
    try:
        # Always work with a safe copy
        working_history = create_safe_history_copy(history)
        
        # Ensure we have an empty assistant response to fill
        if not working_history or working_history[-1].get("role") != "assistant":
            working_history.append({"role": "assistant", "content": ""})
        
        # Stream chunks and build response
        for chunk in streamer:
            if chunk and isinstance(chunk, str):
                # Update the assistant's response
                working_history[-1]["content"] += chunk
                
                # Create a fresh, validated copy for yielding
                yield_history = create_safe_history_copy(working_history)
                print(f"🔄 Yielding {len(yield_history)} messages")
                yield yield_history

    except Exception as e:
        print(f"❌ Streaming error: {e}")
        # Create error response with safe format
        error_history = create_safe_history_copy(history or [])
        if not error_history or error_history[-1].get("role") != "assistant":
            error_history.append({"role": "assistant", "content": ""})
        
        error_history[-1]["content"] = f"❌ Streaming error: {str(e)[:100]}..."
        yield error_history


def handle_chat_error(error: Exception, history: ChatHistory) -> ChatHistory:
    """
    Handle chat errors with user-friendly messages.
    
    Args:
        error: Exception that occurred
        history: Current chat history
        
    Returns:
        Updated history with error message
    """
    print(f"❌ Chat function error: {error}")
    
    # Ensure history is properly formatted
    if not isinstance(history, list):
        history = []
    
    # Determine error type and provide helpful message
    error_message = "❌ An error occurred. "
    error_str = str(error).lower()
    
    if "memory" in error_str:
        error_message += "Memory limit reached. Try starting a new conversation."
    elif "token" in error_str or "length" in error_str:
        error_message += "Message too long. Please try a shorter message."
    elif "compile" in error_str:
        error_message += "NPU compilation issue. Check NPUW configuration."
    elif "timeout" in error_str:
        error_message += "Generation timed out. Try a simpler request."
    elif "device" in error_str:
        error_message += "Device error. NPU may not be available."
    else:
        error_message += f"Details: {str(error)[:100]}..."
    
    # Normalize history to ensure proper format
    updated_history = normalize_history(history)
    
    # Add error to history in the correct format
    if (not updated_history or 
        not isinstance(updated_history[-1], dict) or 
        updated_history[-1].get("role") != "assistant" or 
        updated_history[-1].get("content")):
        # If no history or last message has content, add new exchange
        updated_history.append({"role": "assistant", "content": error_message})
    else:
        # Update the empty bot response
        updated_history[-1]["content"] = error_message
    
    return updated_history


def should_use_agent(message: str) -> bool:
    """
    Determine if the user message should be processed by the agent system.
    
    Args:
        message: User input message
        
    Returns:
        True if agent should be used, False for regular chat
    """
    # Keywords that suggest tool usage
    agent_keywords = [
        'calculate', 'compute', 'math', 'equation', 'solve',
        'what time', 'what date', 'today', 'tomorrow', 'yesterday', 
        'search', 'look up', 'find information',
        'analyze text', 'word count', 'character count',
        'tool', 'function', 'help me with'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in agent_keywords)


def enhanced_llm_chat(message: str, history: ChatHistory, generation_params: Dict[str, Any] = None) -> Iterator[ChatHistory]:
    """
    Enhanced chat function with comprehensive Phi-3 optimization and RAG support.
    This version includes the fix for the Gradio streaming data format error.
    """
    request_start_time = time.time()
    streaming_metrics.start_request()

    # Step 1: Normalize the incoming history to the required List[Dict] format
    # This correctly handles the List[List] format from the Gradio component
    if history is None:
        history = []
    normalized_history = normalize_history(history)

    try:
        # Step 2: Prepare and validate the user's input
        processed_message, was_truncated, updated_history = prepare_chat_input(message, normalized_history)
        
        # Early return for empty messages
        if not processed_message.strip():
            import json
            print("RETURNING (empty message):\n", json.dumps(updated_history, indent=2, default=str))
            yield updated_history
            return

        # If the message was truncated, yield the warning and pause
        if was_truncated:
            import json
            print("RETURNING (truncated):\n", json.dumps(updated_history, indent=2, default=str))
            yield updated_history
            time.sleep(0.5)

        # Step 3: Decide whether to use the ReAct agent or regular chat
        agent = get_agent()
        use_agent = AGENT_AVAILABLE and agent and should_use_agent(processed_message)

        if use_agent:
            print(f"🤖 Using agent processing for: {processed_message[:50]}...")
            try:
                # Agent processing is not streamed, so we yield the final result once
                agent_response = agent.process_with_tools(processed_message, generation_params)
                updated_history[-1]["content"] = agent_response
                import json
                print("RETURNING (agent):\n", json.dumps(updated_history, indent=2, default=str))
                yield updated_history
                return
            except Exception as e:
                print(f"⚠️ Agent processing failed, falling back to regular chat: {e}")

        # Step 4: RAG Context Retrieval for regular chat
        rag_context = rag_system.retrieve_context(processed_message)
        if rag_context:
            augmented_message = f"""Based on the following context from uploaded documents, please answer the user's question. If the context doesn't contain relevant information, please indicate that and provide a general response.

Context:
{rag_context}

Question: {processed_message}"""
            processed_message = augmented_message
            print(f"📚 Using RAG context: {len(rag_context)} characters from documents")
        
        # Step 5: Set up the streamer and generation thread
        def metrics_callback(metric_name: str, value):
            streaming_metrics.update_metric(metric_name, value)
        
        streamer = EnhancedLLMStreamer(tokenizer, metrics_callback)

        def generation_worker():
            success = execute_generation(processed_message, streamer, generation_params)
            elapsed_time = time.time() - request_start_time
            streaming_metrics.end_request(success, elapsed_time)
        
        generation_thread = threading.Thread(target=generation_worker, daemon=True)
        generation_thread.start()

        # Step 6: *** CORRECTED STREAMING LOGIC ***
        # Yield the initial state with the user message and empty bot response
        import json
        print("RETURNING (initial):\n", json.dumps(updated_history, indent=2, default=str))
        yield updated_history

        # Stream the response tokens, appending to the last message
        for chunk in streamer:
            if chunk and isinstance(chunk, str):
                updated_history[-1]["content"] += chunk
                print("RETURNING (streaming):\n", json.dumps(updated_history, indent=2, default=str))
                yield updated_history # Yield the mutated, complete history object each time

        # Wait for the generation thread to finish
        generation_thread.join(timeout=30.0)
        if generation_thread.is_alive():
            print("⚠️ Generation timeout - thread still running")
        
        print(f"📊 Request complete: {time.time() - request_start_time:.2f}s total")

    except Exception as e:
        elapsed_time = time.time() - request_start_time
        streaming_metrics.end_request(False, elapsed_time)
        # Use the already normalized history for error reporting
        error_history = handle_chat_error(e, normalized_history)
        import json
        print("RETURNING (error):\n", json.dumps(error_history, indent=2, default=str))
        yield error_history


def initialize_globals(pipeline, tokenizer_instance):
    """Initialize global pipeline and tokenizer instances"""
    global pipe, tokenizer
    pipe = pipeline
    tokenizer = tokenizer_instance
    
    # Initialize agent system if available
    if AGENT_AVAILABLE:
        try:
            from .agent import initialize_agent
            initialize_agent(pipeline, tokenizer_instance)
            print("✅ Agent system initialized with LLM pipeline")
        except Exception as e:
            print(f"⚠️ Agent initialization failed: {e}")