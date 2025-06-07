"""Base provider abstract class for unified LLM interface."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Any, Iterator, Optional, Tuple
from .models import ChatResponse, ChatStreamResponse
from .utils import extract_function_schema, validate_messages, execute_function_call
from .exceptions import ValidationError, ToolExecutionError
import json


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, model_id: str, tools: Optional[List[Callable]] = None, **kwargs):
        """Initialize provider with model and configuration.
        
        Args:
            model_id: Model identifier for the provider
            tools: Optional list of Python functions to use as tools
            **kwargs: All parameters (common and provider-specific) set once
                     Common params: temperature, max_tokens, top_p, etc.
                     Provider-specific params: top_k, repetition_penalty, etc.
        """
        self.model_id = model_id
        self.tools = tools or []
        self.tool_schemas = {}
        self.tools_by_name = {}
        
        # Store all configuration parameters
        self.config = kwargs
        
        # Process tools during initialization
        if self.tools:
            self._process_tools()
    
    def _process_tools(self) -> None:
        """Process tools and build schemas."""
        for tool in self.tools:
            if not callable(tool):
                raise ValidationError(f"Tool {tool} must be callable")
            
            schema = extract_function_schema(tool)
            tool_name = schema["name"]
            
            # Store tool and schema
            self.tools_by_name[tool_name] = tool
            self.tool_schemas[tool_name] = schema
    
    def chat(self, messages: List[Dict[str, Any]], enable_reasoning: bool = False) -> ChatResponse:
        """Synchronous chat completion with optional reasoning.
        
        Args:
            messages: List of message dictionaries in OpenAI-compatible format
            enable_reasoning: Whether to enable reasoning content extraction
            
        Returns:
            ChatResponse containing the completion result
            
        Raises:
            ValidationError: If messages format is invalid
            ProviderError: If provider API call fails
        """
        # Validate input messages
        validate_messages(messages)
        
        # Prepare request in provider-specific format
        request = self._prepare_request(messages, enable_reasoning=enable_reasoning)
        
        # Execute the request
        response = self._execute_request(request)
        
        # Parse response and handle tool calls if needed
        parsed_response = self._parse_response(response, enable_reasoning=enable_reasoning)
        
        # Check if response contains tool calls
        if hasattr(parsed_response, 'tool_calls') and parsed_response.tool_calls:
            # Execute tools and continue conversation
            return self._handle_tool_calls(messages, parsed_response, enable_reasoning=enable_reasoning)
        
        return parsed_response
    
    def chat_stream(self, messages: List[Dict[str, Any]], enable_reasoning: bool = False) -> Iterator[ChatStreamResponse]:
        """Streaming chat completion with optional reasoning.
        
        Args:
            messages: List of message dictionaries in OpenAI-compatible format
            enable_reasoning: Whether to enable reasoning content extraction
            
        Yields:
            ChatStreamResponse chunks
            
        Raises:
            ValidationError: If messages format is invalid
            ProviderError: If provider API call fails
        """
        # Validate input messages
        validate_messages(messages)
        
        # Prepare streaming request
        request = self._prepare_request(messages, stream=True, enable_reasoning=enable_reasoning)
        
        # Execute streaming request
        for chunk in self._execute_stream_request(request):
            parsed_chunk = self._parse_stream_response(chunk, enable_reasoning=enable_reasoning)
            yield parsed_chunk
    
    def _handle_tool_calls(self, original_messages: List[Dict[str, Any]], 
                          assistant_response: ChatResponse, enable_reasoning: bool = False) -> ChatResponse:
        """Handle tool execution and continue conversation.
        
        Args:
            original_messages: Original conversation messages
            assistant_response: Assistant response containing tool calls
            enable_reasoning: Whether reasoning is enabled
            
        Returns:
            Final ChatResponse after tool execution
        """
        # Build conversation with tool calls
        conversation = original_messages.copy()
        
        # Add assistant message with tool calls
        assistant_msg = {
            "role": "assistant",
            "content": assistant_response.content
        }
        
        if hasattr(assistant_response, 'tool_calls') and assistant_response.tool_calls:
            assistant_msg["tool_calls"] = assistant_response.tool_calls
        
        conversation.append(assistant_msg)
        
        # Execute each tool call
        for tool_call in getattr(assistant_response, 'tool_calls', []):
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", "{}")
            tool_id = tool_call.get("id")
            
            if tool_name not in self.tools_by_name:
                result = f"Error: Tool '{tool_name}' not found"
            else:
                try:
                    tool_func = self.tools_by_name[tool_name]
                    result = execute_function_call(tool_func, tool_args)
                except ToolExecutionError as e:
                    result = f"Error executing {tool_name}: {e}"
            
            # Add tool result to conversation
            tool_result_msg = {
                "role": "tool",
                "content": result,
                "tool_call_id": tool_id
            }
            conversation.append(tool_result_msg)
        
        # Get final response from provider
        return self.chat(conversation, enable_reasoning=enable_reasoning)
    
    @abstractmethod
    def _prepare_request(self, messages: List[Dict[str, Any]], stream: bool = False, enable_reasoning: bool = False) -> Dict[str, Any]:
        """Convert unified format to provider-specific format.
        
        Args:
            messages: Messages in unified format
            stream: Whether this is for streaming
            enable_reasoning: Whether reasoning is enabled
            
        Returns:
            Provider-specific request dictionary
        """
        pass
    
    @abstractmethod
    def _execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute provider-specific API call.
        
        Args:
            request: Provider-specific request
            
        Returns:
            Provider-specific response
            
        Raises:
            ProviderError: If API call fails
        """
        pass
    
    @abstractmethod
    def _execute_stream_request(self, request: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Execute provider-specific streaming API call.
        
        Args:
            request: Provider-specific request
            
        Yields:
            Provider-specific response chunks
            
        Raises:
            ProviderError: If API call fails
        """
        pass
    
    @abstractmethod
    def _parse_response(self, response: Dict[str, Any], enable_reasoning: bool = False) -> ChatResponse:
        """Convert provider response to unified format.
        
        Args:
            response: Provider-specific response
            enable_reasoning: Whether reasoning is enabled
            
        Returns:
            ChatResponse in unified format
        """
        pass
    
    @abstractmethod
    def _parse_stream_response(self, chunk: Dict[str, Any], enable_reasoning: bool = False) -> ChatStreamResponse:
        """Convert provider stream chunk to unified format.
        
        Args:
            chunk: Provider-specific response chunk
            enable_reasoning: Whether reasoning is enabled
            
        Returns:
            ChatStreamResponse in unified format
        """
        pass
    
    @abstractmethod
    def _construct_tools(self, functions: List[Callable]) -> List[Dict[str, Any]]:
        """Convert functions to provider-specific tool schema.
        
        Args:
            functions: List of Python functions
            
        Returns:
            Provider-specific tool schema
        """
        pass
    
    @abstractmethod
    def _extract_reasoning_content(self, response: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """Extract content and reasoning from provider response.
        
        Args:
            response: Provider-specific response
            
        Returns:
            Tuple of (final_content, reasoning_content)
        """
        pass 