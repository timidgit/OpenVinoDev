"""
Agentic Architecture Module (Phase 3.3)
=======================================

Function-calling agent system that transforms the chat application from a passive 
chatbot into an active AI agent capable of using tools and performing complex tasks.

This implements the ReAct (Reasoning + Acting) pattern with OpenVINO GenAI as the 
reasoning engine and LangChain as the agent framework.
"""

import json
import re
import math
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass

# Agent framework imports with fallback
try:
    from langchain_core.tools import Tool
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    AGENT_AVAILABLE = True
    print("✅ Agent framework loaded successfully")
except ImportError as e:
    print(f"⚠️ Agent framework not available: {e}")
    print("📝 Install with: pip install langchain-core langchain-experimental")
    AGENT_AVAILABLE = False


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    result: str
    error: Optional[str] = None


class AgentTools:
    """Collection of tools available to the AI agent"""
    
    @staticmethod
    def calculator(expression: str) -> ToolResult:
        """
        Perform mathematical calculations safely.
        
        Args:
            expression: Mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)")
            
        Returns:
            ToolResult with calculation result or error
        """
        try:
            # Sanitize input - only allow safe mathematical operations
            safe_chars = set('0123456789+-*/()., ')
            safe_functions = ['abs', 'ceil', 'floor', 'sqrt', 'pow', 'log', 'sin', 'cos', 'tan']
            
            # Replace safe function names
            sanitized = expression.lower()
            for func in safe_functions:
                sanitized = sanitized.replace(func, f'math.{func}')
            
            # Check for dangerous patterns
            dangerous = ['import', 'exec', 'eval', '__', 'open', 'file', 'input']
            if any(d in sanitized.lower() for d in dangerous):
                return ToolResult(False, "", "Expression contains unsafe operations")
            
            # Evaluate using math module in restricted environment
            import math
            allowed_names = {
                'math': math,
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'len': len
            }
            
            result = eval(sanitized, {"__builtins__": {}}, allowed_names)
            return ToolResult(True, str(result))
            
        except Exception as e:
            return ToolResult(False, "", f"Calculation error: {str(e)}")
    
    @staticmethod
    def datetime_info(query: str = "") -> ToolResult:
        """
        Get current date, time, or calculate date differences.
        
        Args:
            query: Optional specific query like "tomorrow", "next week", etc.
            
        Returns:
            ToolResult with date/time information
        """
        try:
            now = datetime.now()
            
            if not query or query.lower() in ["now", "current", "today"]:
                result = f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
            
            elif "tomorrow" in query.lower():
                tomorrow = now + timedelta(days=1)
                result = f"Tomorrow's date: {tomorrow.strftime('%Y-%m-%d')}"
            
            elif "yesterday" in query.lower():
                yesterday = now - timedelta(days=1)
                result = f"Yesterday's date: {yesterday.strftime('%Y-%m-%d')}"
            
            elif "next week" in query.lower():
                next_week = now + timedelta(weeks=1)
                result = f"Next week: {next_week.strftime('%Y-%m-%d')}"
            
            elif "week" in query.lower() and "ago" in query.lower():
                last_week = now - timedelta(weeks=1)
                result = f"One week ago: {last_week.strftime('%Y-%m-%d')}"
            
            else:
                result = f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}\nDay of week: {now.strftime('%A')}"
            
            return ToolResult(True, result)
            
        except Exception as e:
            return ToolResult(False, "", f"DateTime error: {str(e)}")
    
    @staticmethod
    def web_search_mock(query: str) -> ToolResult:
        """
        Mock web search tool (placeholder for security).
        In production, this would connect to a search API.
        
        Args:
            query: Search query string
            
        Returns:
            ToolResult with mock search results
        """
        try:
            # This is a mock implementation for security reasons
            # In production, you would integrate with DuckDuckGo API, Google Custom Search, etc.
            
            mock_results = [
                f"Search results for: '{query}'",
                "• [Mock Result 1] This would be a real web search result",
                "• [Mock Result 2] Integration with search APIs available",
                "• [Mock Result 3] Enable by configuring search provider API keys",
                "",
                "Note: This is a mock tool. Configure real search API to enable live results."
            ]
            
            return ToolResult(True, "\n".join(mock_results))
            
        except Exception as e:
            return ToolResult(False, "", f"Search error: {str(e)}")
    
    @staticmethod
    def text_analysis(text: str, analysis_type: str = "summary") -> ToolResult:
        """
        Analyze text content (word count, character count, basic analysis).
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis ("summary", "count", "readability")
            
        Returns:
            ToolResult with text analysis
        """
        try:
            if not text:
                return ToolResult(False, "", "No text provided for analysis")
            
            # Basic analysis metrics
            word_count = len(text.split())
            char_count = len(text)
            char_count_no_spaces = len(text.replace(' ', ''))
            sentence_count = len([s for s in text.split('.') if s.strip()])
            paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
            
            if analysis_type.lower() == "count":
                result = f"Word count: {word_count}\nCharacter count: {char_count}\nCharacters (no spaces): {char_count_no_spaces}\nSentences: {sentence_count}\nParagraphs: {paragraph_count}"
            
            elif analysis_type.lower() == "readability":
                # Simple readability metrics
                avg_words_per_sentence = word_count / max(sentence_count, 1)
                avg_chars_per_word = char_count_no_spaces / max(word_count, 1)
                
                result = f"Readability Analysis:\n• Average words per sentence: {avg_words_per_sentence:.1f}\n• Average characters per word: {avg_chars_per_word:.1f}\n• Text complexity: {'Simple' if avg_words_per_sentence < 15 else 'Moderate' if avg_words_per_sentence < 25 else 'Complex'}"
            
            else:  # summary
                result = f"Text Analysis Summary:\n• Words: {word_count}\n• Characters: {char_count}\n• Sentences: {sentence_count}\n• Paragraphs: {paragraph_count}\n• Average sentence length: {word_count / max(sentence_count, 1):.1f} words"
            
            return ToolResult(True, result)
            
        except Exception as e:
            return ToolResult(False, "", f"Analysis error: {str(e)}")


class ReActAgent:
    """ReAct (Reasoning + Acting) Agent using OpenVINO GenAI"""
    
    def __init__(self, llm_pipeline, tokenizer):
        """
        Initialize ReAct agent with LLM pipeline and available tools.
        
        Args:
            llm_pipeline: OpenVINO GenAI pipeline instance
            tokenizer: Tokenizer instance
        """
        self.llm = llm_pipeline
        self.tokenizer = tokenizer
        self.available = AGENT_AVAILABLE
        
        # Define available tools
        self.tools = {
            "calculator": {
                "function": AgentTools.calculator,
                "description": "Perform mathematical calculations. Input: mathematical expression (e.g., '2+2', 'sqrt(16)', 'sin(3.14/2)')",
                "parameters": "expression (string): Mathematical expression to evaluate"
            },
            "datetime": {
                "function": AgentTools.datetime_info,
                "description": "Get current date/time or calculate date differences. Input: query like 'now', 'tomorrow', 'next week'",
                "parameters": "query (string): Date/time query or leave empty for current datetime"
            },
            "web_search": {
                "function": AgentTools.web_search_mock,
                "description": "Search the web for information (mock implementation). Input: search query string",
                "parameters": "query (string): What to search for"
            },
            "text_analysis": {
                "function": AgentTools.text_analysis,
                "description": "Analyze text content (word count, readability, etc.). Input: text and analysis type",
                "parameters": "text (string): Text to analyze, analysis_type (string): 'summary', 'count', or 'readability'"
            }
        }
        
        # ReAct prompt template
        self.system_prompt = self._create_system_prompt()
        
        if self.available:
            print(f"✅ ReAct agent initialized with {len(self.tools)} tools")
        else:
            print("⚠️ Agent functionality limited - install langchain-core for full capabilities")
    
    def _create_system_prompt(self) -> str:
        """Create the ReAct system prompt with tool descriptions"""
        
        tools_description = "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])
        
        return f"""You are an AI assistant with access to tools. Use the ReAct (Reasoning + Acting) approach to solve problems.

Available Tools:
{tools_description}

When you need to use a tool, format your response EXACTLY like this:
```
Thought: I need to use a tool to help with this request.
Action: tool_name
Action Input: input_for_tool
```

After receiving tool results, continue reasoning and provide a final answer.

Example:
User: What's 25 * 4?
Assistant: I need to calculate this multiplication.

Thought: I need to calculate 25 * 4.
Action: calculator  
Action Input: 25 * 4

[Tool Result: 100]

The result of 25 * 4 is 100.

Always think step by step and use tools when they can help provide accurate information."""

    def _parse_action(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Parse action from LLM response.
        
        Args:
            text: LLM response text
            
        Returns:
            Tuple of (tool_name, input) if action found, None otherwise
        """
        # Look for Action: tool_name pattern
        action_match = re.search(r'Action:\s*(\w+)', text, re.IGNORECASE)
        if not action_match:
            return None
        
        tool_name = action_match.group(1).lower()
        
        # Look for Action Input: input pattern
        input_match = re.search(r'Action Input:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
        if not input_match:
            return None
        
        tool_input = input_match.group(1).strip()
        
        return (tool_name, tool_input)
    
    def _execute_tool(self, tool_name: str, tool_input: str) -> ToolResult:
        """
        Execute a tool with given input.
        
        Args:
            tool_name: Name of tool to execute
            tool_input: Input parameters for the tool
            
        Returns:
            ToolResult with execution result
        """
        if tool_name not in self.tools:
            return ToolResult(False, "", f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")
        
        try:
            tool_func = self.tools[tool_name]["function"]
            
            # Handle different tool signatures
            if tool_name == "text_analysis" and "," in tool_input:
                # Split text and analysis_type
                parts = [p.strip() for p in tool_input.split(",", 1)]
                if len(parts) == 2:
                    return tool_func(parts[0], parts[1])
                else:
                    return tool_func(tool_input)
            else:
                return tool_func(tool_input)
                
        except Exception as e:
            return ToolResult(False, "", f"Tool execution error: {str(e)}")
    
    def process_with_tools(self, user_message: str, generation_params: Dict[str, Any] = None) -> str:
        """
        Process user message using ReAct pattern with tools.
        
        Args:
            user_message: User's input message
            generation_params: Optional generation parameters
            
        Returns:
            Final response after tool usage (if needed)
        """
        if not self.available:
            return f"🤖 Basic Response: {user_message}\n\n⚠️ Agent tools not available. Install langchain-core for full capabilities."
        
        max_iterations = 5  # Prevent infinite loops
        conversation_history = [
            f"System: {self.system_prompt}",
            f"User: {user_message}"
        ]
        
        for iteration in range(max_iterations):
            # Generate response with current conversation context
            full_context = "\n\n".join(conversation_history)
            
            try:
                # Start a fresh chat session for agent reasoning
                self.llm.start_chat(self.system_prompt)
                
                # Generate response
                from .chat import create_phi3_generation_config
                gen_config = create_phi3_generation_config(generation_params)
                
                response = ""
                for chunk in self.llm.generate(full_context, gen_config):
                    response += chunk
                
                self.llm.finish_chat()
                
            except Exception as e:
                return f"❌ Agent processing error: {str(e)}"
            
            # Check if response contains an action
            action_result = self._parse_action(response)
            
            if action_result is None:
                # No action found, this is the final response
                return response
            
            tool_name, tool_input = action_result
            
            # Execute the tool
            print(f"🔧 Executing tool: {tool_name} with input: {tool_input}")
            tool_result = self._execute_tool(tool_name, tool_input)
            
            # Add tool result to conversation
            if tool_result.success:
                conversation_history.append(f"Assistant: {response}")
                conversation_history.append(f"Tool Result: {tool_result.result}")
            else:
                error_msg = tool_result.error or "Tool execution failed"
                conversation_history.append(f"Assistant: {response}")
                conversation_history.append(f"Tool Error: {error_msg}")
        
        # If we've exceeded max iterations, return what we have
        return f"⚠️ Reasoning process exceeded maximum iterations. Last response:\n\n{response}"
    
    def get_available_tools(self) -> Dict[str, str]:
        """Get list of available tools with descriptions"""
        return {
            name: info["description"] 
            for name, info in self.tools.items()
        }


# Global agent instance (initialized by main.py)
react_agent: Optional[ReActAgent] = None


def initialize_agent(llm_pipeline, tokenizer):
    """Initialize the global agent instance"""
    global react_agent
    react_agent = ReActAgent(llm_pipeline, tokenizer)
    return react_agent


def get_agent() -> Optional[ReActAgent]:
    """Get the global agent instance"""
    return react_agent