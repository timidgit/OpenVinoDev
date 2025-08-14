"""
Unit tests for Agent system (Phase 3.3)
========================================

Tests the agentic architecture with function-calling capabilities.
"""

import pytest
from unittest.mock import Mock, patch
from app.agent import AgentTools, ReActAgent, ToolResult, should_use_agent


class TestAgentTools:
    """Test cases for individual agent tools"""
    
    def test_calculator_basic(self):
        """Test basic calculator functionality"""
        result = AgentTools.calculator("2 + 2")
        assert result.success is True
        assert result.result == "4"
    
    def test_calculator_advanced(self):
        """Test advanced math functions"""
        result = AgentTools.calculator("sqrt(16)")
        assert result.success is True
        assert result.result == "4.0"
    
    def test_calculator_security(self):
        """Test calculator security filtering"""
        result = AgentTools.calculator("import os")
        assert result.success is False
        assert "unsafe" in result.error
    
    def test_datetime_current(self):
        """Test current datetime functionality"""
        result = AgentTools.datetime_info("now")
        assert result.success is True
        assert "date and time" in result.result.lower()
    
    def test_datetime_tomorrow(self):
        """Test tomorrow date calculation"""
        result = AgentTools.datetime_info("tomorrow")
        assert result.success is True
        assert "Tomorrow's date" in result.result
    
    def test_web_search_mock(self):
        """Test mock web search functionality"""
        result = AgentTools.web_search_mock("test query")
        assert result.success is True
        assert "Search results for: 'test query'" in result.result
        assert "mock" in result.result.lower()
    
    def test_text_analysis_count(self):
        """Test text analysis word counting"""
        result = AgentTools.text_analysis("Hello world", "count")
        assert result.success is True
        assert "Word count: 2" in result.result
        assert "Character count:" in result.result
    
    def test_text_analysis_summary(self):
        """Test text analysis summary"""
        result = AgentTools.text_analysis("The quick brown fox jumps over the lazy dog.", "summary")
        assert result.success is True
        assert "Words: 9" in result.result
        assert "Sentences:" in result.result


class TestAgentDecisionLogic:
    """Test cases for agent decision logic"""
    
    def test_should_use_agent_calculation(self):
        """Test agent detection for calculation requests"""
        assert should_use_agent("What's 2 + 2?") is True
        assert should_use_agent("Calculate the square root of 16") is True
        assert should_use_agent("Help me solve this math problem") is True
    
    def test_should_use_agent_datetime(self):
        """Test agent detection for datetime requests"""
        assert should_use_agent("What time is it?") is True
        assert should_use_agent("What's today's date?") is True
        assert should_use_agent("What's tomorrow?") is True
    
    def test_should_use_agent_search(self):
        """Test agent detection for search requests"""
        assert should_use_agent("Search for information about AI") is True
        assert should_use_agent("Look up the weather") is True
        assert should_use_agent("Find information about quantum computing") is True
    
    def test_should_use_agent_text_analysis(self):
        """Test agent detection for text analysis"""
        assert should_use_agent("Analyze this text for word count") is True
        assert should_use_agent("How many characters are in this sentence?") is True
    
    def test_should_not_use_agent_regular_chat(self):
        """Test that regular chat doesn't trigger agent"""
        assert should_use_agent("Hello, how are you?") is False
        assert should_use_agent("Tell me about machine learning") is False
        assert should_use_agent("Write a poem about nature") is False


class TestReActAgent:
    """Test cases for ReAct agent system"""
    
    @patch('app.agent.AGENT_AVAILABLE', True)
    def test_agent_initialization(self):
        """Test agent initialization with mocked dependencies"""
        mock_llm = Mock()
        mock_tokenizer = Mock()
        
        agent = ReActAgent(mock_llm, mock_tokenizer)
        
        assert agent.llm == mock_llm
        assert agent.tokenizer == mock_tokenizer
        assert len(agent.tools) == 4  # calculator, datetime, web_search, text_analysis
        assert agent.available is True
    
    def test_parse_action_valid(self):
        """Test parsing valid action from LLM response"""
        mock_llm = Mock()
        mock_tokenizer = Mock()
        agent = ReActAgent(mock_llm, mock_tokenizer)
        
        response = """I need to calculate this.
        
        Thought: I should use the calculator tool.
        Action: calculator
        Action Input: 2 + 2
        """
        
        result = agent._parse_action(response)
        assert result is not None
        assert result[0] == "calculator"
        assert result[1] == "2 + 2"
    
    def test_parse_action_invalid(self):
        """Test parsing response without action"""
        mock_llm = Mock()
        mock_tokenizer = Mock()
        agent = ReActAgent(mock_llm, mock_tokenizer)
        
        response = "This is just a regular response without any action."
        
        result = agent._parse_action(response)
        assert result is None
    
    def test_execute_tool_calculator(self):
        """Test tool execution for calculator"""
        mock_llm = Mock()
        mock_tokenizer = Mock()
        agent = ReActAgent(mock_llm, mock_tokenizer)
        
        result = agent._execute_tool("calculator", "5 * 6")
        assert result.success is True
        assert result.result == "30"
    
    def test_execute_tool_invalid(self):
        """Test execution of invalid tool"""
        mock_llm = Mock()
        mock_tokenizer = Mock()
        agent = ReActAgent(mock_llm, mock_tokenizer)
        
        result = agent._execute_tool("nonexistent_tool", "input")
        assert result.success is False
        assert "not found" in result.error
    
    def test_get_available_tools(self):
        """Test getting available tools list"""
        mock_llm = Mock()
        mock_tokenizer = Mock()
        agent = ReActAgent(mock_llm, mock_tokenizer)
        
        tools = agent.get_available_tools()
        assert isinstance(tools, dict)
        assert "calculator" in tools
        assert "datetime" in tools
        assert "web_search" in tools
        assert "text_analysis" in tools


if __name__ == "__main__":
    pytest.main([__file__])