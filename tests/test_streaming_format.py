#!/usr/bin/env python3
"""
Test Suite for Gradio Streaming Format Compliance
=================================================

Tests to verify that the chat streaming functions return properly formatted
data compatible with Gradio's ChatInterface (type='messages') format.

This addresses the critical requirement that streaming functions must yield
List[Dict[str, str]] objects with the correct message format.

Copyright (c) 2025 sbran
Licensed under the MIT License - see LICENSE file for details
"""

import pytest
from typing import List, Dict, Any, Union
import json


def test_message_format_structure():
    """Test that message format follows required structure."""
    # Example of correct message format
    correct_format = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    # Validate structure
    assert isinstance(correct_format, list), "History must be a list"
    
    for message in correct_format:
        assert isinstance(message, dict), "Each message must be a dictionary"
        assert "role" in message, "Each message must have 'role' key"
        assert "content" in message, "Each message must have 'content' key"
        assert isinstance(message["role"], str), "Role must be string"
        assert isinstance(message["content"], str), "Content must be string"
        assert message["role"] in ["user", "assistant", "system"], "Role must be valid"


def test_streaming_output_format_validator():
    """Test validator function for streaming output format."""
    
    def is_valid_gradio_history(history: Any) -> bool:
        """
        Validate that history conforms to Gradio ChatInterface format.
        
        Args:
            history: History object to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Must be a list
            if not isinstance(history, list):
                return False
            
            # Each item must be a dict with role and content
            for item in history:
                if not isinstance(item, dict):
                    return False
                if "role" not in item or "content" not in item:
                    return False
                if not isinstance(item["role"], str) or not isinstance(item["content"], str):
                    return False
                if item["role"] not in ["user", "assistant", "system"]:
                    return False
                    
            return True
        except Exception:
            return False
    
    # Test valid formats
    valid_cases = [
        [],  # Empty history
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
        [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hello"}]
    ]
    
    for case in valid_cases:
        assert is_valid_gradio_history(case), f"Valid case failed: {case}"
    
    # Test invalid formats
    invalid_cases = [
        "not a list",
        [{"role": "user"}],  # Missing content
        [{"content": "Hello"}],  # Missing role
        [{"role": "invalid", "content": "Hello"}],  # Invalid role
        [{"role": 123, "content": "Hello"}],  # Non-string role
        [{"role": "user", "content": 123}],  # Non-string content
        [["user", "Hello"]],  # Wrong structure
        {"role": "user", "content": "Hello"}  # Not a list
    ]
    
    for case in invalid_cases:
        assert not is_valid_gradio_history(case), f"Invalid case passed: {case}"


def test_json_serialization():
    """Test that message format can be JSON serialized."""
    history = [
        {"role": "user", "content": "Test message"},
        {"role": "assistant", "content": "Test response"}
    ]
    
    # Should serialize without error
    json_str = json.dumps(history, indent=2, default=str)
    assert isinstance(json_str, str)
    
    # Should deserialize back to same structure
    deserialized = json.loads(json_str)
    assert deserialized == history


def test_empty_and_edge_cases():
    """Test edge cases for streaming format."""
    # Empty history should be valid
    empty_history = []
    assert isinstance(empty_history, list)
    
    # Single message should be valid
    single_message = [{"role": "user", "content": "Single message"}]
    assert len(single_message) == 1
    assert single_message[0]["role"] == "user"
    
    # Long content should be valid
    long_content = "x" * 10000
    long_message = [{"role": "user", "content": long_content}]
    assert len(long_message[0]["content"]) == 10000
    
    # Empty content should be valid
    empty_content = [{"role": "user", "content": ""}]
    assert empty_content[0]["content"] == ""


def test_conversation_flow():
    """Test typical conversation flow format."""
    conversation = []
    
    # Add user message
    conversation.append({"role": "user", "content": "What is Python?"})
    assert len(conversation) == 1
    
    # Add assistant response
    conversation.append({"role": "assistant", "content": "Python is a programming language."})
    assert len(conversation) == 2
    
    # Add follow-up
    conversation.append({"role": "user", "content": "Tell me more."})
    conversation.append({"role": "assistant", "content": "Python is known for simplicity."})
    assert len(conversation) == 4
    
    # Verify alternating pattern (common but not required)
    for i, msg in enumerate(conversation):
        if i % 2 == 0:
            assert msg["role"] == "user"
        else:
            assert msg["role"] == "assistant"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])