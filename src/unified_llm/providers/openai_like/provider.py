"""OpenAI-compatible provider for unified LLM interface."""

import os
import json
import uuid
import re
from typing import Dict, List, Any, Iterator, Optional, Callable, Tuple
import httpx
from ...base import BaseProvider
from ...models import ChatResponse, ChatStreamResponse
from ...exceptions import ProviderError, ConfigurationError
from ...utils import extract_function_schema


class OpenAILike(BaseProvider):
    """Provider for OpenAI-compatible endpoints (OpenAI, vLLM, Ollama, etc.).
    
    Supports both official OpenAI API and any compatible endpoints like:
    - vLLM server
    - Ollama with OpenAI compatibility
    - LocalAI
    - text-generation-webui with OpenAI extension
    """
    
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
        
        # Initialize streaming reasoning state
        self._in_reasoning_mode = False
        
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
    
    def _standardize_tool_calls(self, raw_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardize OpenAI tool calls to consistent format.
        
        OpenAI API returns tool calls in this format:
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "calculate",
                "arguments": "{\"expression\": \"25 * 4\"}"
            }
        }
        
        We standardize to:
        {
            "id": "call_abc123",
            "name": "calculate", 
            "arguments": "{\"expression\": \"25 * 4\"}"
        }
        
        Args:
            raw_tool_calls: Tool calls from OpenAI API
            
        Returns:
            Standardized tool call format
        """
        if not raw_tool_calls:
            return []
        
        standardized = []
        for tool_call in raw_tool_calls:
            if isinstance(tool_call, dict):
                if "function" in tool_call:
                    # Full OpenAI format - convert to standardized
                    standardized.append({
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"]
                    })
                else:
                    # Already standardized format - pass through
                    standardized.append(tool_call)
            else:
                # Handle potential non-dict formats
                standardized.append(tool_call)
        
        return standardized
    
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
            ProviderError: If streaming API call fails
        """
        try:
            with self.client.stream(
                'POST',
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
                
                # Process streaming response
                for line in response.iter_lines():
                    if line:
                        # Handle both bytes and string cases
                        if isinstance(line, bytes):
                            line = line.decode('utf-8')
                        
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                yield chunk
                            except json.JSONDecodeError:
                                continue
                                
        except httpx.RequestError as e:
            raise ProviderError(f"Streaming request failed: {e}", provider="openai_like")
    
    def _extract_reasoning_content(self, response: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """Extract content and reasoning from OpenAI-compatible response.
        
        Handles multiple reasoning formats:
        1. Native reasoning_content field (vLLM)
        2. Pattern-based extraction from content (<think>, <reasoning> tags)
        3. OpenAI o1-style reasoning tokens in usage
        4. Thinking field (some providers)
        
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
            (r'<analysis>(.*?)</analysis>', re.DOTALL),
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
            # but note the token count in metadata
            return full_content, None
        
        # No reasoning found
        return full_content, None
    
    def _parse_response(self, response: Dict[str, Any], enable_reasoning: bool = False) -> ChatResponse:
        """Parse OpenAI API response to unified format.
        
        Args:
            response: OpenAI-compatible response
            enable_reasoning: Whether reasoning is enabled
            
        Returns:
            ChatResponse in unified format with standardized tool calls
        """
        # Extract main content
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        
        # Extract tool calls and standardize them internally
        raw_tool_calls = message.get("tool_calls")
        standardized_tool_calls = self._standardize_tool_calls(raw_tool_calls or [])
        
        # Handle reasoning content extraction
        reasoning_content = None
        reasoning_tokens = None
        
        if enable_reasoning:
            content, reasoning_content = self._extract_reasoning_content(response)            
            # Extract reasoning token count if available (OpenAI o1)
            usage = response.get('usage', {})
            completion_tokens_details = usage.get('completion_tokens_details', {})
            reasoning_tokens = completion_tokens_details.get('reasoning_tokens')
        
        # Build metadata
        metadata = {
            "model": response.get("model"),
            "usage": response.get("usage", {}),
            "finish_reason": response["choices"][0].get("finish_reason"),
            "system_fingerprint": response.get("system_fingerprint"),
            "provider": "openai_like",
            "endpoint": getattr(self, 'base_url', 'unknown')
        }
        
        return ChatResponse(
            content=content,
            reasoning_content=reasoning_content,
            reasoning_tokens=reasoning_tokens,
            tool_calls=standardized_tool_calls if standardized_tool_calls else None,
            metadata=metadata
        )
    
    def _parse_stream_response(self, chunk: Dict[str, Any], enable_reasoning: bool = False) -> ChatStreamResponse:
        """Parse OpenAI streaming response chunk to unified format.
        
        Args:
            chunk: OpenAI-compatible response chunk
            enable_reasoning: Whether reasoning is enabled
            
        Returns:
            ChatStreamResponse in unified format with standardized tool calls
        """
        if not chunk.get("choices"):
            return ChatStreamResponse(delta="", metadata={"raw_chunk": chunk})
        
        choice = chunk["choices"][0]
        delta = choice.get("delta", {})
        
        # Extract content delta
        content_delta = delta.get("content") or ""
        
        # Extract tool calls and standardize them internally
        raw_tool_calls = delta.get("tool_calls")
        standardized_tool_calls = self._standardize_tool_calls(raw_tool_calls or [])
        
        # Handle reasoning in streaming
        reasoning_delta = None
        is_reasoning_complete = False
        
        if enable_reasoning:
            # Check for reasoning_content delta (vLLM)
            reasoning_delta = delta.get('reasoning_content')
            
            # Check for thinking delta (some providers)
            if reasoning_delta is None:
                reasoning_delta = delta.get('thinking')
            
            # For pattern-based reasoning, detect start/end transitions
            if content_delta and reasoning_delta is None:
                full_delta = content_delta
                
                # Check for reasoning START (opening tags)
                opening_tags = ['<think>', '<thinking>', '<reasoning>', '<analysis>']
                if any(tag in full_delta for tag in opening_tags):
                    self._in_reasoning_mode = True
                    print(f"🧠 Reasoning started detected in chunk")  # Debug
                
                # Check for reasoning END (closing tags)
                closing_tags = ['</think>', '</thinking>', '</reasoning>', '</analysis>']
                if any(tag in full_delta for tag in closing_tags):
                    self._in_reasoning_mode = False
                    is_reasoning_complete = True
                    print(f"✅ Reasoning completed detected in chunk")  # Debug
                
                # Route content based on current reasoning state
                if self._in_reasoning_mode:
                    # We're in reasoning mode - this delta is reasoning content
                    reasoning_delta = content_delta
                    content_delta = ""  # Don't emit as content
                else:
                    # We're in normal mode - emit as regular content
                    # content_delta remains as-is
                    pass
        
        # Check if stream is complete
        is_complete = choice.get("finish_reason") is not None
        
        # Build metadata
        metadata = {
            "model": chunk.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "raw_chunk": chunk,
            "provider": "openai_like"
        }
        
        # Add usage information if available (usually only in final chunk)
        if 'usage' in chunk:
            metadata['usage'] = chunk['usage']
        
        return ChatStreamResponse(
            delta=content_delta,
            reasoning_delta=reasoning_delta,
            is_reasoning_complete=is_reasoning_complete,
            is_complete=is_complete,
            tool_calls=standardized_tool_calls if standardized_tool_calls else None,
            metadata=metadata
        )
    
    def _construct_tools(self, functions: List[Callable]) -> List[Dict[str, Any]]:
        """Convert functions to OpenAI tool schema.
        
        Args:
            functions: List of Python functions
            
        Returns:
            OpenAI-compatible tool schema for API requests
        """
        tools = []
        for func in functions:
            schema = self.tool_schemas[func.__name__]
            
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