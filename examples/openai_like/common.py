#!/usr/bin/env python3
"""
Common utilities and tools for OpenAI-like provider examples.

This module contains shared functions and configurations used across
all examples to avoid code duplication.
"""

import sys
import os

# Add src to path for importing unified_llm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.unified_llm import OpenAILike, ToolExecutor


def calculate(expression: str) -> float:
    """Calculate a mathematical expression safely.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        The result of the calculation
    """
    try:
        # Only allow basic math operations and numbers
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        
        result = eval(expression)
        return float(result)
    except:
        raise ValueError(f"Cannot evaluate expression: {expression}")


def get_weather(location: str) -> str:
    """Get weather information for a location.
    
    Args:
        location: The city or location to get weather for
    
    Returns:
        Weather description string
    """
    # Mock weather data
    weather_data = {
        "san francisco": "Foggy, 16°C",
        "new york": "Sunny, 22°C", 
        "london": "Rainy, 12°C",
        "tokyo": "Cloudy, 18°C",
        "paris": "Sunny, 24°C",
        "sydney": "Windy, 19°C"
    }
    
    location_key = location.lower()
    return weather_data.get(location_key, f"Weather data not available for {location}")


def search_web(query: str) -> str:
    """Mock web search function.
    
    Args:
        query: Search query string
        
    Returns:
        Mock search results
    """
    mock_results = {
        "python": "Python is a high-level programming language known for its simplicity.",
        "ai": "Artificial Intelligence is the simulation of human intelligence in machines.",
        "weather": "Weather refers to atmospheric conditions at a specific time and place.",
        "news": "Breaking: Scientists discover new method for clean energy production.",
    }
    
    query_key = query.lower()
    for key, result in mock_results.items():
        if key in query_key:
            return result
    
    return f"Search results for '{query}': No specific results found in mock database."


def get_default_provider(**kwargs) -> OpenAILike:
    """Get a default configured OpenAI-like provider.
    
    Args:
        **kwargs: Override default configuration
        
    Returns:
        Configured OpenAI-like provider
    """
    default_config = {
        "model_id": "qwen3:4b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "fake-key",
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    # Override with provided kwargs
    default_config.update(kwargs)
    
    return OpenAILike(**default_config)


def get_default_executor() -> ToolExecutor:
    """Get a default tool executor with common tools.
    
    Returns:
        ToolExecutor with calculate, get_weather, and search_web tools
    """
    return ToolExecutor(tools=[calculate, get_weather, search_web])


def print_section(title: str, width: int = 60):
    """Print a formatted section header.
    
    Args:
        title: Section title
        width: Total width of the header
    """
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_subsection(title: str, width: int = 40):
    """Print a formatted subsection header.
    
    Args:
        title: Subsection title
        width: Total width of the header
    """
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


def execute_tool_calls(executor: ToolExecutor, tool_calls: list, verbose: bool = True) -> list:
    """Execute a list of tool calls and return formatted results.
    
    Args:
        executor: ToolExecutor instance
        tool_calls: List of tool call dictionaries
        verbose: Whether to print execution details
        
    Returns:
        List of tool result messages
    """
    tool_results = []
    
    for tool_call in tool_calls:
        if verbose:
            print(f"   🔧 Calling {tool_call['name']}({tool_call['arguments']})")
        
        result = executor.execute(tool_call)
        
        # Format as tool result message for conversation
        tool_result = {
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call["id"]
        }
        tool_results.append(tool_result)
        
        if verbose:
            print(f"   ✅ Result: {result}")
    
    return tool_results


def handle_error(e: Exception, context: str = ""):
    """Handle and display errors gracefully.
    
    Args:
        e: Exception that occurred
        context: Additional context about where the error occurred
    """
    error_msg = f"Error{f' in {context}' if context else ''}: {e}"
    print(f"❌ {error_msg}")
    
    # Provide helpful hints for common errors
    if "connection" in str(e).lower():
        print("💡 Hint: Make sure your local LLM server (Ollama/vLLM) is running")
        print("   For Ollama: ollama serve")
        print("   For vLLM: python -m vllm.entrypoints.openai.api_server --model <model_name>")
    elif "api_key" in str(e).lower():
        print("💡 Hint: Set OPENAI_API_KEY environment variable for OpenAI API") 