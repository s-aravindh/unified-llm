"""OpenAI-compatible provider for unified LLM interface."""

import os
import json
import uuid
import re
from typing import Dict, List, Any, Iterator, Optional, Callable, Tuple
import httpx
from .base import BaseProvider
from .models import ChatResponse, ChatStreamResponse
from .exceptions import ProviderError, ConfigurationError
from .utils import extract_function_schema


class OpenAILike(BaseProvider):
    """Provider for OpenAI-compatible endpoints (vLLM, Ollama, etc.)."""
    
    # Common parameters supported across OpenAI-compatible providers
    COMMON_PARAMS = {
        'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 
        'presence_penalty', 'stop', 'seed'
    }
    
    def __init__(self, model_id: str, tools: Optional[List[Callable]] = None, **kwargs):
        """Initialize OpenAI-compatible provider.
        
        Args:
            model_id: Model identifier
            tools: Optional list of Python functions to use as tools
            **kwargs: All parameters including:
                Common params: temperature, max_tokens, top_p, frequency_penalty, etc.
                Provider-specific: top_k, repetition_penalty, min_p, typical_p, etc.
                Connection: base_url, api_key, timeout
        """
        # Extract connection parameters
        self.base_url = kwargs.pop('base_url', os.getenv('OPENAI_LIKE_BASE_URL', 'http://localhost:8000/v1'))
        self.api_key = kwargs.pop('api_key', os.getenv('OPENAI_LIKE_API_KEY', 'dummy-key'))
        self.timeout = kwargs.pop('timeout', 30)
        
        # Ensure base_url ends with /v1
        if not self.base_url.endswith('/v1'):
            if self.base_url.endswith('/'):
                self.base_url += 'v1'
            else:
                self.base_url += '/v1'
        
        # Separate common parameters from provider-specific parameters
        self.common_params = {}
        self.provider_params = {}
        
        for key, value in kwargs.items():
            if key in self.COMMON_PARAMS:
                self.common_params[key] = value
            else:
                self.provider_params[key] = value
        
        # Set defaults for common parameters if not provided
        self.common_params.setdefault('temperature', 0.7)
        self.common_params.setdefault('max_tokens', 1000)
        
        # Call parent constructor with all parameters
        super().__init__(model_id, tools, **kwargs)
        
        # Initialize httpx client
        self.client = httpx.Client(
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            },
            timeout=self.timeout
        )
        
        # Test connection
        self._test_connection()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
    
    def close(self):
        """Close the HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
    
    def _test_connection(self) -> None:
        """Test connection to the API endpoint."""
        try:
            response = self.client.get(f"{self.base_url}/models")
            if response.status_code != 200:
                raise ProviderError(
                    f"Failed to connect to OpenAI-like endpoint: {response.status_code}",
                    provider="openai_like",
                    status_code=response.status_code
                )
        except httpx.RequestError as e:
            raise ConfigurationError(f"Cannot connect to OpenAI-like endpoint at {self.base_url}: {e}")
    
    def _prepare_request(self, messages: List[Dict[str, Any]], stream: bool = False, enable_reasoning: bool = False) -> Dict[str, Any]:
        """Convert unified format to OpenAI-compatible format.
        
        Args:
            messages: Messages in unified format
            stream: Whether this is for streaming
            enable_reasoning: Whether reasoning is enabled
            
        Returns:
            OpenAI-compatible request dictionary
        """
        # Convert multimodal content if present
        converted_messages = []
        for message in messages:
            converted_msg = message.copy()
            
            # Handle multimodal content
            if isinstance(message.get('content'), list):
                content_parts = []
                for part in message['content']:
                    if part['type'] == 'text':
                        content_parts.append({
                            'type': 'text',
                            'text': part['text']
                        })
                    elif part['type'] == 'image':
                        # Convert to OpenAI image format
                        content_parts.append({
                            'type': 'image_url',
                            'image_url': {
                                'url': f"data:image/jpeg;base64,{part['image_data']}"
                            }
                        })
                converted_msg['content'] = content_parts
            
            converted_messages.append(converted_msg)
        
        # Build request with configured parameters
        request = {
            'model': self.model_id,
            'messages': converted_messages,
            'stream': stream,
            **self.common_params,      # Add all common parameters
            **self.provider_params     # Add all provider-specific parameters
        }
        
        # Add tools if available
        if self.tools:
            request['tools'] = self._construct_tools(self.tools)
            request['tool_choice'] = 'auto'
        
        return request
    
    def _execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute OpenAI-compatible API call.
        
        Args:
            request: OpenAI-compatible request
            
        Returns:
            OpenAI-compatible response
            
        Raises:
            ProviderError: If API call fails
        """
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=request
            )
            
            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get('error', {}).get('message', error_detail)
                except:
                    pass
                
                raise ProviderError(
                    f"API request failed: {error_detail}",
                    provider="openai_like",
                    status_code=response.status_code
                )
            
            return response.json()
            
        except httpx.RequestError as e:
            raise ProviderError(f"Request failed: {e}", provider="openai_like")
    
    def _execute_stream_request(self, request: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Execute OpenAI-compatible streaming API call.
        
        Args:
            request: OpenAI-compatible request
            
        Yields:
            OpenAI-compatible response chunks
            
        Raises:
            ProviderError: If API call fails
        """
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=request
            ) as response:
                
                if response.status_code != 200:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = error_json.get('error', {}).get('message', error_detail)
                    except:
                        pass
                    
                    raise ProviderError(
                        f"Streaming API request failed: {error_detail}",
                        provider="openai_like",
                        status_code=response.status_code
                    )
                
                for line in response.iter_lines():
                    if line:
                        line_str = line.strip()
                        if line_str.startswith('data: '):
                            data = line_str[6:]  # Remove 'data: ' prefix
                            if data.strip() == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                yield chunk
                            except json.JSONDecodeError:
                                continue
                            
        except httpx.RequestError as e:
            raise ProviderError(f"Streaming request failed: {e}", provider="openai_like")
    
    def _extract_reasoning_content(self, response: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """Extract content and reasoning from provider response.
        
        For OpenAI-like providers, this handles multiple reasoning formats:
        1. Native reasoning_content field (vLLM)
        2. Pattern-based extraction from content (<think>, <reasoning> tags)
        3. OpenAI o1-style reasoning tokens in usage
        
        Args:
            response: Provider-specific response
            
        Returns:
            Tuple of (final_content, reasoning_content)
        """
        if 'choices' not in response or not response['choices']:
            return "", None
        
        choice = response['choices'][0]
        message = choice.get('message', {})
        
        # Method 1: Check for native reasoning_content field (vLLM)
        if 'reasoning_content' in message:
            content = message.get('content', '')
            reasoning = message.get('reasoning_content')
            return content, reasoning
        
        # Method 2: Check for thinking field (some providers)
        if 'thinking' in message:
            content = message.get('content', '')
            reasoning = message.get('thinking')
            return content, reasoning
        
        # Method 3: Pattern-based extraction from content
        full_content = message.get('content', '')
        if not full_content:
            return "", None
        
        # Common reasoning patterns
        patterns = [
            (r'<think>(.*?)</think>', re.DOTALL),
            (r'<thinking>(.*?)</thinking>', re.DOTALL), 
            (r'<reasoning>(.*?)</reasoning>', re.DOTALL),
        ]
        
        for pattern, flags in patterns:
            match = re.search(pattern, full_content, flags)
            if match:
                reasoning_content = match.group(1).strip()
                # Remove reasoning section from final content
                final_content = re.sub(pattern, '', full_content, flags=flags).strip()
                return final_content, reasoning_content
        
        # Method 4: Check for reasoning tokens in usage (OpenAI o1)
        usage = response.get('usage', {})
        completion_tokens_details = usage.get('completion_tokens_details', {})
        reasoning_tokens = completion_tokens_details.get('reasoning_tokens')
        
        if reasoning_tokens and reasoning_tokens > 0:
            # For OpenAI o1, reasoning is hidden, so we return None for reasoning_content
            # but note the token count
            return full_content, None
        
        # No reasoning found
        return full_content, None
    
    def _parse_response(self, response: Dict[str, Any], enable_reasoning: bool = False) -> ChatResponse:
        """Convert OpenAI-compatible response to unified format.
        
        Args:
            response: OpenAI-compatible response
            enable_reasoning: Whether reasoning is enabled
            
        Returns:
            ChatResponse in unified format
        """
        if 'choices' not in response or not response['choices']:
            raise ProviderError("Invalid response format: missing choices", provider="openai_like")
        
        choice = response['choices'][0]
        message = choice.get('message', {})
        
        # Extract content and reasoning
        if enable_reasoning:
            content, reasoning_content = self._extract_reasoning_content(response)
        else:
            content = message.get('content', '')
            reasoning_content = None
        
        # Get tool calls
        tool_calls = message.get('tool_calls', [])
        
        # Get reasoning token count if available
        reasoning_tokens = None
        if enable_reasoning:
            usage = response.get('usage', {})
            completion_tokens_details = usage.get('completion_tokens_details', {})
            reasoning_tokens = completion_tokens_details.get('reasoning_tokens')
        
        # Build metadata with provider-specific information
        metadata = {}
        
        # Usage information
        if 'usage' in response:
            metadata['usage'] = response['usage']
        
        # Model information
        if 'model' in response:
            metadata['model'] = response['model']
        
        # Response metadata
        if 'id' in response:
            metadata['response_id'] = response['id']
        
        if 'created' in response:
            metadata['created'] = response['created']
        
        # Choice metadata
        if 'finish_reason' in choice:
            metadata['finish_reason'] = choice['finish_reason']
        
        if 'index' in choice:
            metadata['choice_index'] = choice['index']
        
        # System fingerprint (for OpenAI compatibility)
        if 'system_fingerprint' in response:
            metadata['system_fingerprint'] = response['system_fingerprint']
        
        # Provider identifier and configuration used
        metadata['provider'] = 'openai_like'
        metadata['endpoint'] = self.base_url
        metadata['common_params'] = self.common_params.copy()
        metadata['provider_params'] = self.provider_params.copy()
        
        chat_response = ChatResponse(
            content=content, 
            reasoning_content=reasoning_content,
            reasoning_tokens=reasoning_tokens,
            metadata=metadata
        )
        
        # Add tool calls if present
        if tool_calls:
            chat_response.tool_calls = tool_calls
        
        return chat_response
    
    def _parse_stream_response(self, chunk: Dict[str, Any], enable_reasoning: bool = False) -> ChatStreamResponse:
        """Convert OpenAI-compatible stream response chunk to unified format.
        
        Args:
            chunk: OpenAI-compatible response chunk
            enable_reasoning: Whether reasoning is enabled
            
        Returns:
            ChatStreamResponse in unified format
        """
        if 'choices' not in chunk or not chunk['choices']:
            return ChatStreamResponse(delta="", is_complete=False)
        
        choice = chunk['choices'][0]
        delta = choice.get('delta', {})
        
        content_delta = delta.get('content', '') or ''
        reasoning_delta = None
        is_reasoning_complete = False
        
        if enable_reasoning:
            # Check for reasoning_content delta (vLLM)
            reasoning_delta = delta.get('reasoning_content')
            
            # Check for thinking delta (some providers)
            if reasoning_delta is None:
                reasoning_delta = delta.get('thinking')
            
            # For pattern-based reasoning, we need to track state
            # This is simplified - in practice, you might need more sophisticated tracking
            if content_delta and reasoning_delta is None:
                # Check if we're transitioning from reasoning to content
                full_delta = content_delta
                if '</think>' in full_delta or '</thinking>' in full_delta or '</reasoning>' in full_delta:
                    is_reasoning_complete = True
                    # Extract reasoning and content from delta
                    patterns = [
                        (r'<think>(.*?)</think>', re.DOTALL),
                        (r'<thinking>(.*?)</thinking>', re.DOTALL), 
                        (r'<reasoning>(.*?)</reasoning>', re.DOTALL),
                    ]
                    
                    for pattern, flags in patterns:
                        if re.search(pattern, full_delta, flags):
                            match = re.search(pattern, full_delta, flags)
                            if match:
                                reasoning_delta = match.group(1).strip()
                                content_delta = re.sub(pattern, '', full_delta, flags=flags).strip()
                                break
        
        # Check if stream is complete
        finish_reason = choice.get('finish_reason')
        is_complete = finish_reason is not None
        
        # Build metadata for stream chunk
        metadata = {}
        
        # Chunk identification
        if 'id' in chunk:
            metadata['chunk_id'] = chunk['id']
        
        if 'created' in chunk:
            metadata['created'] = chunk['created']
        
        # Model information
        if 'model' in chunk:
            metadata['model'] = chunk['model']
        
        # Choice metadata
        if 'index' in choice:
            metadata['choice_index'] = choice['index']
        
        if finish_reason:
            metadata['finish_reason'] = finish_reason
        
        # Usage information (usually only in final chunk)
        if 'usage' in chunk:
            metadata['usage'] = chunk['usage']
        
        # Provider information
        metadata['provider'] = 'openai_like'
        metadata['endpoint'] = self.base_url
        
        return ChatStreamResponse(
            delta=content_delta,
            reasoning_delta=reasoning_delta,
            is_reasoning_complete=is_reasoning_complete,
            is_complete=is_complete,
            metadata=metadata
        )
    
    def _construct_tools(self, functions: List[Callable]) -> List[Dict[str, Any]]:
        """Convert functions to OpenAI-compatible tool format.
        
        Args:
            functions: List of Python functions
            
        Returns:
            OpenAI-compatible tool schema list
        """
        tools = []
        for func in functions:
            schema = extract_function_schema(func)
            
            tool = {
                'type': 'function',
                'function': {
                    'name': schema['name'],
                    'description': schema['description'],
                    'parameters': schema['parameters']
                }
            }
            tools.append(tool)
        
        return tools 