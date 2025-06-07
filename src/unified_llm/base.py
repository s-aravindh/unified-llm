"""Base provider abstract class for unified LLM interface."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Any, Iterator, Optional, Tuple
from .models import ChatResponse, ChatStreamResponse
from .utils import extract_function_schema, validate_messages
from .exceptions import ValidationError
import json


class BaseProvider(ABC):
    """Abstract base class for all LLM providers.
    
    Pure interface design: Providers handle format translation and return tool calls
    as data. They do NOT execute tools automatically. Applications control all tool
    execution and conversation flow.
    """
    
    def __init__(self, model_id: str, tools: Optional[List[Callable]] = None, **kwargs):
        """Initialize provider with model and configuration.
        
        Args:
            model_id: Model identifier for the provider
            tools: Optional list of Python functions for schema generation ONLY
                   (tools are NOT executed automatically by the provider)
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
        
        # Process tools during initialization (for schema generation only)
        if self.tools:
            self._process_tools()
    
    def _process_tools(self) -> None:
        """Process tools and build schemas for provider use."""
        for tool in self.tools:
            if not callable(tool):
                raise ValidationError(f"Tool {tool} must be callable")
            
            schema = extract_function_schema(tool)
            tool_name = schema["name"]
            
            # Store tool and schema for provider schema generation
            self.tools_by_name[tool_name] = tool
            self.tool_schemas[tool_name] = schema
    
    def chat(self, messages: List[Dict[str, Any]], enable_reasoning: bool = False) -> ChatResponse:
        """Synchronous chat completion with optional reasoning.
        
        Pure interface: Returns tool calls as data without executing them.
        Applications are responsible for tool execution and conversation management.
        
        Args:
            messages: List of message dictionaries in OpenAI-compatible format
            enable_reasoning: Whether to enable reasoning content extraction
            
        Returns:
            ChatResponse containing the completion result with tool_calls as data
            
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
        
        # Parse response and return (with tool calls as data, not executed)
        return self._parse_response(response, enable_reasoning=enable_reasoning)
    
    def chat_stream(self, messages: List[Dict[str, Any]], enable_reasoning: bool = False) -> Iterator[ChatStreamResponse]:
        """Streaming chat completion with optional reasoning.
        
        Pure interface: Returns tool calls as data without executing them.
        Applications are responsible for tool execution and conversation management.
        
        Args:
            messages: List of message dictionaries in OpenAI-compatible format
            enable_reasoning: Whether to enable reasoning content extraction
            
        Yields:
            ChatStreamResponse chunks with tool_calls as data
            
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
            ChatResponse in unified format with tool_calls as data
        """
        pass
    
    @abstractmethod
    def _parse_stream_response(self, chunk: Dict[str, Any], enable_reasoning: bool = False) -> ChatStreamResponse:
        """Convert provider stream chunk to unified format.
        
        Args:
            chunk: Provider-specific response chunk
            enable_reasoning: Whether reasoning is enabled
            
        Returns:
            ChatStreamResponse in unified format with tool_calls as data
        """
        pass
    
    @abstractmethod
    def _construct_tools(self, functions: List[Callable]) -> List[Dict[str, Any]]:
        """Convert functions to provider-specific tool schema.
        
        Args:
            functions: List of Python functions
            
        Returns:
            Provider-specific tool schema for API requests
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