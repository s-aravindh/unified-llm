"""Tool execution utility for unified LLM interface."""

from typing import List, Callable, Dict, Any
from .utils import execute_function_call
from .exceptions import ToolExecutionError


class ToolExecutor:
    """Utility class for executing individual tool calls with error handling.
    
    This is a separate utility that applications can optionally use to execute
    tool calls returned by providers. It provides safe execution with proper
    error handling and validation.
    """
    
    def __init__(self, tools: List[Callable]):
        """Initialize with available tools.
        
        Args:
            tools: List of callable functions that can be executed
        """
        self.tools_by_name = {}
        
        # Build tool registry
        for tool in tools:
            if not callable(tool):
                raise ValueError(f"Tool {tool} must be callable")
            self.tools_by_name[tool.__name__] = tool
    
    def execute(self, tool_call: Dict[str, Any]) -> str:
        """Execute a single tool call with comprehensive error handling.
        
        Args:
            tool_call: Tool call dictionary with 'name', 'arguments', 'id' fields
            
        Returns:
            String result of tool execution (success result or error message)
            
        Examples:
            >>> executor = ToolExecutor(tools=[get_weather])
            >>> result = executor.execute({
            ...     "name": "get_weather",
            ...     "arguments": '{"location": "New York"}',
            ...     "id": "call_123"
            ... })
            >>> print(result)  # "Weather in New York: sunny, 22°C"
        """
        try:
            # Validate tool call structure
            if not isinstance(tool_call, dict):
                return "Error: Tool call must be a dictionary"
            
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", "{}")
            tool_id = tool_call.get("id")
            
            # Validate required fields
            if not tool_name:
                return "Error: Missing tool name in tool call"
            
            if not tool_id:
                return "Error: Missing tool call ID"
            
            # Check if tool exists
            if tool_name not in self.tools_by_name:
                available_tools = ", ".join(self.tools_by_name.keys())
                return f"Error: Tool '{tool_name}' not found. Available tools: {available_tools}"
            
            # Execute the tool
            tool_func = self.tools_by_name[tool_name]
            result = execute_function_call(tool_func, tool_args)
            
            # Ensure result is a string
            return str(result)
            
        except ToolExecutionError as e:
            return f"Tool execution error: {e}"
        except Exception as e:
            return f"Unexpected error executing tool '{tool_name}': {e}"
    
    def execute_all(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple tool calls and return formatted results.
        
        Args:
            tool_calls: List of tool call dictionaries
            
        Returns:
            List of tool result message dictionaries ready for conversation
            
        Examples:
            >>> executor = ToolExecutor(tools=[get_weather, calculate])
            >>> tool_calls = [
            ...     {"name": "get_weather", "arguments": '{"location": "Tokyo"}', "id": "call_1"},
            ...     {"name": "calculate", "arguments": '{"expression": "2+2"}', "id": "call_2"}
            ... ]
            >>> results = executor.execute_all(tool_calls)
            >>> # Returns formatted tool result messages
        """
        results = []
        
        for tool_call in tool_calls:
            result_content = self.execute(tool_call)
            tool_id = tool_call.get("id", "unknown")
            
            # Format as tool result message
            tool_result = {
                "role": "tool",
                "content": result_content,
                "tool_call_id": tool_id
            }
            results.append(tool_result)
        
        return results
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names.
        
        Returns:
            List of tool names that can be executed
        """
        return list(self.tools_by_name.keys())
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if a specific tool is available.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if tool is available, False otherwise
        """
        return tool_name in self.tools_by_name
    
    def validate_tool_call(self, tool_call: Dict[str, Any]) -> List[str]:
        """Validate a tool call without executing it.
        
        Args:
            tool_call: Tool call dictionary to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not isinstance(tool_call, dict):
            errors.append("Tool call must be a dictionary")
            return errors
        
        # Check required fields
        if not tool_call.get("name"):
            errors.append("Missing tool name")
        
        if not tool_call.get("id"):
            errors.append("Missing tool call ID")
        
        # Check if tool exists
        tool_name = tool_call.get("name")
        if tool_name and tool_name not in self.tools_by_name:
            available = ", ".join(self.tools_by_name.keys())
            errors.append(f"Tool '{tool_name}' not found. Available: {available}")
        
        return errors 