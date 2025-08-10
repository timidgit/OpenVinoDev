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
from .streamer import EnhancedQwen3Streamer, streaming_metrics

# Type definitions
ChatMessage = TypedDict('ChatMessage', {'role': str, 'content': str})
ChatHistory = List[ChatMessage]

# RAG system imports with fallback
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    RAG_AVAILABLE = True
    print("✅ RAG dependencies loaded successfully")
except ImportError as e:
    print(f"⚠️ RAG dependencies not available: {e}")
    print("📝 Install with: pip install langchain faiss-cpu sentence-transformers")
    RAG_AVAILABLE = False


class DocumentRAGSystem:
    """Retrieval-Augmented Generation system for document processing"""
    
    def __init__(self):
        """Initialize RAG system with fallback handling"""
        self.vector_store = None
        self.embeddings = None
        self.text_splitter = None
        self.processed_docs_count = 0
        self.available = RAG_AVAILABLE
        
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
                
                print("✅ RAG system initialized successfully")
                
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
            # Read file content with encoding detection
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
            
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
            
            return f"✅ Successfully processed '{file_name}': {len(chunks)} chunks created from {len(text):,} characters. Ready to answer questions about this document."
            
        except Exception as e:
            return f"❌ Error processing '{file_name}': {str(e)}"
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User question to find relevant context for
            k: Number of top chunks to retrieve
            
        Returns:
            Concatenated context from relevant document chunks
        """
        if not self.available or self.vector_store is None:
            return ""
        
        try:
            # Search for relevant documents
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
        """Get current RAG system status"""
        return {
            "Available": self.available,
            "Documents Processed": self.processed_docs_count,
            "Vector Store": "Loaded" if self.vector_store is not None else "Empty",
            "Embedding Model": "all-MiniLM-L6-v2" if self.available else "None"
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


def create_qwen3_generation_config() -> ov_genai.GenerationConfig:
    """
    Create optimized generation configuration for Qwen3 from configuration file.
    
    Returns:
        Configured GenerationConfig with security-conscious defaults
    """
    gen_config = ov_genai.GenerationConfig()
    config = get_config()
    
    # Load generation settings from configuration
    gen_settings = config.get_section("generation")
    
    gen_config.do_sample = gen_settings.get("do_sample", True)
    gen_config.temperature = min(gen_settings.get("temperature", 0.6), 2.0)  # Security: cap temperature
    gen_config.top_p = min(gen_settings.get("top_p", 0.95), 1.0)  # Security: cap top_p
    gen_config.top_k = min(gen_settings.get("top_k", 20), 100)  # Security: reasonable top_k
    gen_config.max_new_tokens = min(gen_settings.get("max_new_tokens", 1024), 2048)  # Security: limit tokens
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
    
    Args:
        message: Raw user input
        history: Current chat history
        
    Returns:
        Tuple of (processed_message, was_truncated, updated_history)
        
    Raises:
        ValueError: If message fails security validation
    """
    # Input validation
    if not message.strip():
        return message, False, history
    
    # Security validation
    is_valid, reason = InputValidator.validate_message(message)
    if not is_valid:
        error_history = history.copy()
        error_history.append({
            "role": "assistant", 
            "content": f"🚫 Message rejected: {reason}. Please try a different message."
        })
        raise ValueError(f"Security validation failed: {reason}")
    
    # Sanitize input
    sanitized_message = InputValidator.sanitize_message(message)
    
    # Process message with smart handling
    processed_message, was_truncated = process_user_message(sanitized_message, history)
    
    # Update history with user message and truncation warning if needed
    updated_history = history.copy()
    
    if was_truncated:
        truncation_warning = {
            "role": "assistant",
            "content": f"⚠️ Your message was truncated from {len(message):,} to {len(processed_message)} characters due to NPU memory limits. Processing the truncated version..."
        }
        updated_history.append({"role": "user", "content": message})
        updated_history.append(truncation_warning)
    else:
        updated_history.append({"role": "user", "content": processed_message})
    
    # Add assistant placeholder
    updated_history.append({"role": "assistant", "content": ""})
    
    return processed_message, was_truncated, updated_history


def execute_generation(processed_message: str, streamer: EnhancedQwen3Streamer) -> bool:
    """
    Execute model generation in a controlled manner.
    
    Args:
        processed_message: Message to generate response for
        streamer: Configured streamer for token processing
        
    Returns:
        True if generation succeeded, False otherwise
    """
    try:
        generation_config = create_qwen3_generation_config()
        pipe.generate(processed_message, generation_config, streamer)
        return True
    except Exception as e:
        print(f"❌ Generation error: {e}")
        # Send error through streamer
        error_msg = f"❌ Generation error: {str(e)[:100]}..."
        streamer.text_queue.put(error_msg)
        streamer.text_queue.put(None)
        return False


def stream_response_to_history(streamer: EnhancedQwen3Streamer, history: ChatHistory) -> Iterator[ChatHistory]:
    """
    Stream model response tokens to chat history.
    
    Args:
        streamer: Active streamer with generation in progress
        history: Chat history to update
        
    Yields:
        Updated history with streaming response
    """
    try:
        for chunk in streamer:
            if chunk:  # Only add non-empty chunks
                history[-1]["content"] += chunk
                yield history
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        history[-1]["content"] = f"❌ Streaming error: {str(e)[:100]}..."
        yield history


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
    
    # Add error to history
    updated_history = history.copy()
    if not updated_history or updated_history[-1]["role"] != "assistant":
        updated_history.append({"role": "assistant", "content": error_message})
    else:
        updated_history[-1]["content"] = error_message
    
    return updated_history


def enhanced_qwen3_chat(message: str, history: ChatHistory) -> Iterator[ChatHistory]:
    """
    Enhanced chat function with comprehensive Qwen3 optimization and RAG support.
    
    This is the main chat processing function that handles user input,
    processes it through the Qwen3 model with optional document context,
    and streams back the response with comprehensive error handling and performance monitoring.
    
    Args:
        message: User input message to process
        history: Current chat conversation history
        
    Yields:
        Updated chat history with streaming response as it's generated
    """
    request_start_time = time.time()
    streaming_metrics.start_request()
    
    try:
        # Step 1: Prepare and validate input
        processed_message, was_truncated, updated_history = prepare_chat_input(message, history)
        
        # Early return for empty messages
        if not processed_message.strip():
            yield updated_history
            return
        
        # Show truncation warning with brief pause
        if was_truncated:
            yield updated_history
            time.sleep(0.5)  # Brief pause for user to see warning
        
        # Step 1.5: RAG Context Retrieval
        rag_context = ""
        if rag_system.available and rag_system.vector_store is not None:
            rag_context = rag_system.retrieve_context(processed_message)
            if rag_context:
                # Augment the message with context
                augmented_message = f"""Based on the following context from uploaded documents, please answer the user's question. If the context doesn't contain relevant information, please indicate that and provide a general response.

Context:
{rag_context}

Question: {processed_message}"""
                processed_message = augmented_message
                print(f"📚 Using RAG context: {len(rag_context)} characters from documents")
        
        # Step 2: Initialize streaming components
        def metrics_callback(metric_name: str, value):
            streaming_metrics.update_metric(metric_name, value)
        
        streamer = EnhancedQwen3Streamer(tokenizer, metrics_callback)
        
        # Step 3: Execute generation in separate thread
        def generation_worker():
            success = execute_generation(processed_message, streamer)
            elapsed_time = time.time() - request_start_time
            streaming_metrics.end_request(success, elapsed_time)
        
        generation_thread = threading.Thread(target=generation_worker, daemon=True)
        generation_thread.start()
        
        # Step 4: Stream response to UI
        yield from stream_response_to_history(streamer, updated_history)
        
        # Step 5: Wait for generation completion with timeout
        generation_thread.join(timeout=30.0)
        if generation_thread.is_alive():
            print("⚠️ Generation timeout - thread still running")
        
        print(f"📊 Request complete: {time.time() - request_start_time:.2f}s total")
        
    except Exception as e:
        elapsed_time = time.time() - request_start_time
        streaming_metrics.end_request(False, elapsed_time)
        error_history = handle_chat_error(e, history)
        yield error_history


def initialize_globals(pipeline, tokenizer_instance):
    """Initialize global pipeline and tokenizer instances"""
    global pipe, tokenizer
    pipe = pipeline
    tokenizer = tokenizer_instance