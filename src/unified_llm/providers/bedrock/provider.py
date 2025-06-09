"""AWS Bedrock provider for unified LLM interface."""

from typing import Dict, List, Any, Iterator, Optional, Callable, Tuple
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from ...base import BaseProvider
from ...models import ChatResponse, ChatStreamResponse
from ...exceptions import ProviderError, ConfigurationError


class Bedrock(BaseProvider):
    """Provider for AWS Bedrock foundation models using the Converse API.
    
    Simple interface to AWS Bedrock that supports any model ID. Features are
    automatically attempted and Bedrock API handles unsupported capabilities.
    
    Features:
    - Flexible authentication (client injection, AWS profiles, default credential chain)
    - Tool calling with automatic schema conversion
    - Reasoning support (when model supports it)
    - Multimodal support (images, documents, video)
    - Streaming responses with proper event handling
    - Cross-region model access
    """
    

    
    def __init__(
        self, 
        model_id: str, 
        tools: Optional[List[Callable]] = None,
        bedrock_client: Optional[boto3.client] = None,
        aws_profile: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        **kwargs
    ):
        """Initialize Bedrock provider.
        
        Args:
            model_id: Bedrock model identifier (e.g., 'anthropic.claude-3-5-sonnet-20240620-v1:0')
            tools: Optional list of Python functions for schema generation
            bedrock_client: Pre-configured boto3 Bedrock client (takes precedence over all other auth)
            aws_profile: AWS profile name for credential resolution
            region_name: AWS region for Bedrock client (default: us-east-1)
            aws_access_key_id: AWS access key ID (alternative to profile/default chain)
            aws_secret_access_key: AWS secret access key (required if access_key_id provided)
            aws_session_token: AWS session token (for temporary credentials)
            **kwargs: Additional configuration parameters:
                Common params: temperature, max_tokens, top_p, etc.
                Bedrock-specific params: top_k, stop_sequences, reasoning_budget_tokens, etc.
        """
        # Store authentication parameters for lazy client initialization
        self._bedrock_client = bedrock_client
        self._aws_profile = aws_profile
        self._region_name = region_name or 'us-east-1'
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token
        
        # Initialize client to None for lazy loading
        self._client = None
        
        # Store Bedrock-specific configuration
        self.reasoning_budget_tokens = kwargs.pop('reasoning_budget_tokens', 2000)
        
        # Detect model capabilities based on model_id patterns
        self._detect_model_capabilities()
        
        # Call parent constructor
        super().__init__(model_id, tools, **kwargs)
    

    def _detect_model_capabilities(self) -> None:
        """Set default model capabilities - let API handle unsupported features."""
        # Default to supporting all features - let Bedrock API return appropriate errors
        # if a specific model doesn't support a feature. This avoids maintenance overhead.
        self.supports_reasoning = True
        self.supports_multimodal = True  
        self.supports_tools = True
    
    @property
    def client(self) -> boto3.client:
        """Get or initialize the Bedrock client with lazy loading."""
        if self._client is None:
            self._client = self._initialize_bedrock_client()
        return self._client
    
    def _initialize_bedrock_client(self) -> boto3.client:
        """Initialize Bedrock client with flexible authentication.
        
        Authentication priority:
        1. Pre-configured client passed in constructor
        2. Explicit AWS credentials (access key + secret key)
        3. AWS profile if specified
        4. Default AWS credential chain
        """
        # Priority 1: Use pre-configured client
        if self._bedrock_client is not None:
            return self._bedrock_client
        
        # Priority 2: Use explicit AWS credentials if provided
        if self._aws_access_key_id:
            return boto3.client(
                'bedrock-runtime',
                region_name=self._region_name,
                aws_access_key_id=self._aws_access_key_id,
                aws_secret_access_key=self._aws_secret_access_key,
                aws_session_token=self._aws_session_token  # Can be None
            )
        
        # Priority 3: Use AWS profile if specified
        if self._aws_profile:
            session = boto3.Session(profile_name=self._aws_profile)
            return session.client('bedrock-runtime', region_name=self._region_name)
        
        # Priority 4: Use default AWS credential chain
        return boto3.client('bedrock-runtime', region_name=self._region_name)
    

    def _standardize_tool_calls(self, raw_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardize Bedrock tool calls to OpenAI-compatible format.
        
        Bedrock uses a different tool call format than OpenAI. This method converts
        from Bedrock's format to the standardized format used by the framework.
        
        Args:
            raw_tool_calls: Tool calls in Bedrock format
            
        Returns:
            Tool calls in standardized OpenAI-compatible format
        """
        standardized = []
        
        for tool_call in raw_tool_calls:
            if 'toolUse' in tool_call:
                tool_use = tool_call['toolUse']
                standardized_call = {
                    "id": tool_use.get('toolUseId', f"call_{len(standardized)}"),
                    "name": tool_use.get('name'),
                    "arguments": tool_use.get('input', {})
                }
                # Convert arguments to JSON string if it's a dict
                if isinstance(standardized_call["arguments"], dict):
                    import json
                    standardized_call["arguments"] = json.dumps(standardized_call["arguments"])
                
                standardized.append(standardized_call)
        
        return standardized
    
    def _prepare_request(
        self, 
        messages: List[Dict[str, Any]], 
        stream: bool = False, 
        enable_reasoning: bool = False
    ) -> Dict[str, Any]:
        """Convert unified format to Bedrock Converse API format.
        
        Args:
            messages: Messages in unified format (OpenAI-compatible)
            stream: Whether this is for streaming
            enable_reasoning: Whether reasoning is enabled
            
        Returns:
            Bedrock Converse API request dictionary
        """
        # Convert messages from unified format to Bedrock format
        bedrock_messages = []
        system_messages = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Handle system messages separately (Bedrock uses separate 'system' field)
            if role == "system":
                system_messages.append({"text": content})
                continue
            
            # Convert message content to Bedrock format
            bedrock_content = self._convert_content_to_bedrock(content)
            
            bedrock_message = {
                "role": role,  # "user" or "assistant"
                "content": bedrock_content
            }
            
            # Handle tool calls in assistant messages
            if role == "assistant" and "tool_calls" in message:
                # Add tool use content blocks for each tool call
                for tool_call in message["tool_calls"]:
                    import json
                    tool_content = {
                        "toolUse": {
                            "toolUseId": tool_call["id"],
                            "name": tool_call["name"],
                            "input": json.loads(tool_call["arguments"]) if isinstance(tool_call["arguments"], str) else tool_call["arguments"]
                        }
                    }
                    bedrock_message["content"].append(tool_content)
            
            # Handle reasoning content in assistant messages
            if role == "assistant" and enable_reasoning and "reasoning" in message:
                reasoning_content = {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": message["reasoning"],
                            "signature": message.get("reasoning_signature")  # Optional signature for multi-turn
                        }
                    }
                }
                bedrock_message["content"].append(reasoning_content)
            
            # Handle tool results in user messages  
            if role == "tool":
                # Convert tool result to Bedrock format
                tool_result_content = {
                    "toolResult": {
                        "toolUseId": message.get("tool_call_id", ""),
                        "content": [{"text": content}],
                        "status": "success"
                    }
                }
                bedrock_message = {
                    "role": "user",  # Tool results are sent as user messages in Bedrock
                    "content": [tool_result_content]
                }
            
            bedrock_messages.append(bedrock_message)
        
        # Build the base request
        request = {
            "modelId": self.model_id,
            "messages": bedrock_messages
        }
        
        # Add system prompt if we have any
        if system_messages:
            request["system"] = system_messages
        
        # Add inference configuration from stored parameters
        inference_config = {}
        
        # Map common parameters to Bedrock inference config
        if hasattr(self, 'config'):
            if 'temperature' in self.config:
                inference_config['temperature'] = self.config['temperature']
            if 'max_tokens' in self.config:
                inference_config['maxTokens'] = self.config['max_tokens']
            if 'top_p' in self.config:
                inference_config['topP'] = self.config['top_p']
            if 'stop_sequences' in self.config:
                inference_config['stopSequences'] = self.config['stop_sequences']
        
        if inference_config:
            request["inferenceConfig"] = inference_config
        
        # Add tools if available
        if self.tools:
            tool_config = self._construct_tools(self.tools)
            request["toolConfig"] = tool_config
        
        # Add additional model-specific parameters (NOT for reasoning - that goes in content)
        additional_fields = {}
        if hasattr(self, 'config'):
            # Add model-specific parameters like top_k for Anthropic models
            if 'top_k' in self.config:
                additional_fields['top_k'] = self.config['top_k']
            # Add any other model-specific parameters
            for key, value in self.config.items():
                if key not in ['temperature', 'max_tokens', 'top_p', 'stop_sequences'] and not key.startswith('_'):
                    additional_fields[key] = value
        
        if additional_fields:
            request["additionalModelRequestFields"] = additional_fields
        

        
        return request
    
    def _convert_content_to_bedrock(self, content) -> List[Dict[str, Any]]:
        """Convert unified content format to Bedrock content format.
        
        Args:
            content: Content in unified format (string or list of content blocks)
            
        Returns:
            List of Bedrock content blocks
        """
        # Handle simple text content
        if isinstance(content, str):
            return [{"text": content}]
        
        # Handle multimodal content (list of content blocks)
        if isinstance(content, list):
            bedrock_content = []
            
            for block in content:
                if block["type"] == "text":
                    bedrock_content.append({"text": block["text"]})
                
                elif block["type"] == "image":
                    # Convert base64 image data to bytes for Bedrock
                    import base64
                    image_data = base64.b64decode(block["image_data"])
                    
                    # Determine image format (default to jpeg)
                    image_format = block.get("format", "jpeg")
                    
                    bedrock_content.append({
                        "image": {
                            "format": image_format,
                            "source": {
                                "bytes": image_data
                            }
                        }
                    })
                
                elif block["type"] == "document":
                    # Convert document content
                    import base64
                    document_data = base64.b64decode(block["document_data"])
                    
                    bedrock_content.append({
                        "document": {
                            "format": block.get("format", "txt"),
                            "name": block.get("name", "document"),
                            "source": {
                                "bytes": document_data
                            }
                        }
                    })
                
                elif block["type"] == "video":
                    # Convert video content
                    import base64
                    video_data = base64.b64decode(block["video_data"])
                    
                    bedrock_content.append({
                        "video": {
                            "format": block.get("format", "mp4"),
                            "source": {
                                "bytes": video_data
                            }
                        }
                    })
            
            return bedrock_content
        
        # Fallback for unknown content types
        return [{"text": str(content)}]
    
    def _execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Bedrock Converse API call.
        
        Args:
            request: Bedrock Converse API request
            
        Returns:
            Bedrock Converse API response
            
        Raises:
            ProviderError: If API call fails
        """
        try:
            response = self.client.converse(**request)
            return response
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            raise ProviderError(
                f"Bedrock API error [{error_code}]: {error_message}",
                provider="bedrock",
                status_code=e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            )
        except Exception as e:
            raise ProviderError(f"Unexpected error calling Bedrock: {str(e)}", provider="bedrock")
    
    def _execute_stream_request(self, request: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Execute Bedrock ConverseStream API call.
        
        Args:
            request: Bedrock ConverseStream API request
            
        Yields:
            Bedrock ConverseStream API response chunks
            
        Raises:
            ProviderError: If streaming API call fails
        """
        try:
            response = self.client.converse_stream(**request)
            
            # Process the event stream
            for event in response.get('stream', []):
                yield event
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            raise ProviderError(
                f"Bedrock streaming API error [{error_code}]: {error_message}",
                provider="bedrock",
                status_code=e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            )
        except Exception as e:
            raise ProviderError(f"Unexpected error calling Bedrock stream: {str(e)}", provider="bedrock")
    
    def _parse_response(
        self, 
        response: Dict[str, Any], 
        enable_reasoning: bool = False
    ) -> ChatResponse:
        """Convert Bedrock response to unified format.
        
        Args:
            response: Bedrock Converse API response
            enable_reasoning: Whether reasoning is enabled
            
        Returns:
            ChatResponse in unified format
        """
        # Extract the main output message
        output = response.get("output", {})
        message = output.get("message", {})
        
        # Extract content and reasoning
        final_content, reasoning_content = self._extract_reasoning_content(response)
        
        # Extract tool calls and standardize them
        raw_tool_calls = []
        content_blocks = message.get("content", [])
        
        for block in content_blocks:
            if "toolUse" in block:
                raw_tool_calls.append(block)
        
        standardized_tool_calls = self._standardize_tool_calls(raw_tool_calls)
        
        # Extract metadata
        usage = response.get("usage", {})
        stop_reason = response.get("stopReason")
        
        metadata = {
            "model": self.model_id,
            "usage": usage,
            "stop_reason": stop_reason,
            "provider": "bedrock",
            "raw_response": response
        }
        
        # Extract reasoning tokens if available
        reasoning_tokens = None
        if reasoning_content and usage:
            # Some models may provide reasoning token count in usage
            reasoning_tokens = usage.get("reasoningTokens")
        
        return ChatResponse(
            content=final_content,
            reasoning_content=reasoning_content,
            reasoning_tokens=reasoning_tokens,
            tool_calls=standardized_tool_calls if standardized_tool_calls else None,
            metadata=metadata
        )
    
    def _parse_stream_response(
        self, 
        chunk: Dict[str, Any], 
        enable_reasoning: bool = False
    ) -> ChatStreamResponse:
        """Convert Bedrock stream chunk to unified format.
        
        Args:
            chunk: Bedrock ConverseStream API response chunk
            enable_reasoning: Whether reasoning is enabled
            
        Returns:
            ChatStreamResponse in unified format
        """
        # Initialize default values
        content_delta = None
        reasoning_delta = None
        tool_call_delta = None
        is_finished = False
        finish_reason = None
        metadata = {}
        
        # Process different event types according to AWS documentation
        if 'messageStart' in chunk:
            # Start of message - no delta content
            metadata['role'] = chunk['messageStart'].get('role')
            
        elif 'contentBlockDelta' in chunk:
            # Content delta - extract text, tool use, or reasoning content
            delta = chunk['contentBlockDelta'].get('delta', {})
            
            if 'text' in delta:
                content_delta = delta['text']
                
            elif 'toolUse' in delta:
                # Tool use delta
                tool_call_delta = {
                    'id': chunk['contentBlockStart'].get('start', {}).get('toolUse', {}).get('toolUseId'),
                    'name': chunk['contentBlockStart'].get('start', {}).get('toolUse', {}).get('name'),
                    'arguments_delta': delta['toolUse'].get('input', '')
                }
                
            elif 'reasoningContent' in delta:
                # Reasoning content delta (Claude 3.7+ models)
                reasoning_delta = delta['reasoningContent'].get('text', '')
                
        elif 'contentBlockStop' in chunk:
            # End of content block - no delta content
            pass
            
        elif 'messageStop' in chunk:
            # End of message
            is_finished = True
            finish_reason = chunk['messageStop'].get('stopReason')
            
        elif 'metadata' in chunk:
            # Final metadata with usage and metrics
            usage = chunk['metadata'].get('usage', {})
            metrics = chunk['metadata'].get('metrics', {})
            
            metadata.update({
                'usage': usage,
                'metrics': metrics,
                'model': self.model_id,
                'provider': 'bedrock'
            })
        
        # Convert to the correct ChatStreamResponse format
        delta = content_delta if content_delta else ""
        is_complete = is_finished
        tool_calls = [tool_call_delta] if tool_call_delta else None
        
        if finish_reason:
            metadata['finish_reason'] = finish_reason
        
        return ChatStreamResponse(
            delta=delta,
            reasoning_delta=reasoning_delta,
            is_reasoning_complete=False,  # Set based on contentBlockStop for reasoning
            is_complete=is_complete,
            tool_calls=tool_calls,
            metadata=metadata
        )
    
    def _construct_tools(self, functions: List[Callable]) -> Dict[str, Any]:
        """Convert functions to Bedrock tool configuration format.
        
        Args:
            functions: List of Python functions
            
        Returns:
            Bedrock tool configuration dictionary
        """
        tools = []
        
        for func in functions:
            # Extract function schema using the utility from base class
            schema = self.tool_schemas.get(func.__name__)
            if not schema:
                continue
            
            # Convert to Bedrock tool format
            bedrock_tool = {
                "toolSpec": {
                    "name": func.__name__,
                    "description": schema.get("description", ""),
                    "inputSchema": {
                        "json": schema.get("parameters", {})
                    }
                }
            }
            
            tools.append(bedrock_tool)
        
        return {
            "tools": tools,
            "toolChoice": {"auto": {}}  # Default to auto tool choice
        }
    
    def _extract_reasoning_content(self, response: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """Extract content and reasoning from Bedrock response.
        
        Args:
            response: Bedrock Converse API response
            
        Returns:
            Tuple of (final_content, reasoning_content)
        """
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        
        final_content_parts = []
        reasoning_content_parts = []
        
        for block in content_blocks:
            # Handle text content
            if "text" in block:
                final_content_parts.append(block["text"])
            
            # Handle reasoning content (Claude 3.7+ models)
            elif "reasoningContent" in block:
                reasoning_text = block["reasoningContent"].get("reasoningText", {})
                reasoning_content_parts.append(reasoning_text.get("text", ""))
        
        # Combine content parts
        final_content = "\n".join(final_content_parts) if final_content_parts else ""
        reasoning_content = "\n".join(reasoning_content_parts) if reasoning_content_parts else None
        
        return final_content, reasoning_content 