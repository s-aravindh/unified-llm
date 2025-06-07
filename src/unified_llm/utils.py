"""Utility functions for unified LLM interface."""

import inspect
import json
from typing import Callable, Dict, List, Any, get_type_hints
from .exceptions import ValidationError


def extract_function_schema(func: Callable) -> Dict[str, Any]:
    """Extract function schema from a Python function for tool use.
    
    Args:
        func: Python function to extract schema from
        
    Returns:
        Dictionary containing function name, description, and parameter schema
    """
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func) or f"Function {func.__name__}"
    
    # Get type hints
    try:
        type_hints = get_type_hints(func)
    except (NameError, AttributeError):
        type_hints = {}
    
    properties = {}
    required = []
    
    for param_name, param in signature.parameters.items():
        if param_name == 'self':
            continue
            
        param_info = {
            "type": "string",  # Default type
            "description": f"Parameter {param_name}"
        }
        
        # Extract type information
        if param_name in type_hints:
            python_type = type_hints[param_name]
            if python_type == str:
                param_info["type"] = "string"
            elif python_type == int:
                param_info["type"] = "integer"
            elif python_type == float:
                param_info["type"] = "number"
            elif python_type == bool:
                param_info["type"] = "boolean"
            elif hasattr(python_type, '__origin__') and python_type.__origin__ == list:
                param_info["type"] = "array"
            elif hasattr(python_type, '__origin__') and python_type.__origin__ == dict:
                param_info["type"] = "object"
        
        properties[param_name] = param_info
        
        # Check if parameter is required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    return {
        "name": func.__name__,
        "description": docstring,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }


def validate_messages(messages: List[Dict[str, Any]]) -> None:
    """Validate message format according to OpenAI-compatible standard.
    
    Args:
        messages: List of message dictionaries
        
    Raises:
        ValidationError: If messages don't conform to expected format
    """
    if not isinstance(messages, list):
        raise ValidationError("Messages must be a list")
    
    if not messages:
        raise ValidationError("Messages list cannot be empty")
    
    valid_roles = {"system", "user", "assistant", "tool"}
    
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValidationError(f"Message {i} must be a dictionary")
        
        if "role" not in message:
            raise ValidationError(f"Message {i} missing 'role' field")
        
        if "content" not in message:
            raise ValidationError(f"Message {i} missing 'content' field")
        
        role = message["role"]
        if role not in valid_roles:
            raise ValidationError(f"Message {i} has invalid role '{role}'. Must be one of: {valid_roles}")
        
        content = message["content"]
        
        # Validate content format
        if isinstance(content, str):
            # Simple text content
            continue
        elif isinstance(content, list):
            # Multimodal content blocks
            for j, block in enumerate(content):
                if not isinstance(block, dict):
                    raise ValidationError(f"Message {i}, content block {j} must be a dictionary")
                
                if "type" not in block:
                    raise ValidationError(f"Message {i}, content block {j} missing 'type' field")
                
                block_type = block["type"]
                if block_type == "text":
                    if "text" not in block:
                        raise ValidationError(f"Message {i}, content block {j} of type 'text' missing 'text' field")
                elif block_type == "image":
                    if "image_data" not in block:
                        raise ValidationError(f"Message {i}, content block {j} of type 'image' missing 'image_data' field")
                else:
                    raise ValidationError(f"Message {i}, content block {j} has unsupported type '{block_type}'")
        else:
            raise ValidationError(f"Message {i} content must be string or list of content blocks")
        
        # Additional validation for tool role
        if role == "tool":
            if "tool_call_id" not in message:
                raise ValidationError(f"Message {i} with role 'tool' missing 'tool_call_id' field")


def parse_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse tool calls from assistant message content.
    
    Args:
        content: Raw content that might contain tool calls
        
    Returns:
        List of parsed tool call dictionaries
    """
    # This is a simplified implementation
    # In practice, you'd parse the actual tool call format from the provider
    return []


def execute_function_call(func: Callable, arguments: str) -> str:
    """Execute a function call with JSON arguments.
    
    Args:
        func: Function to execute
        arguments: JSON string containing function arguments
        
    Returns:
        String result of function execution
        
    Raises:
        ToolExecutionError: If function execution fails
    """
    from .exceptions import ToolExecutionError
    
    try:
        # Parse arguments
        if isinstance(arguments, str):
            args_dict = json.loads(arguments)
        else:
            args_dict = arguments
        
        # Execute function
        result = func(**args_dict)
        
        # Convert result to string
        if isinstance(result, str):
            return result
        else:
            return str(result)
            
    except json.JSONDecodeError as e:
        raise ToolExecutionError(f"Invalid JSON arguments: {e}", tool_name=func.__name__, original_error=e)
    except TypeError as e:
        raise ToolExecutionError(f"Invalid function arguments: {e}", tool_name=func.__name__, original_error=e)
    except Exception as e:
        raise ToolExecutionError(f"Function execution failed: {e}", tool_name=func.__name__, original_error=e) 