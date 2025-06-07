# Unified LLM Interface

A **provider-agnostic LLM interface** that provides seamless switching between different LLM providers while maintaining a consistent API surface. Currently supports OpenAI-compatible endpoints (vLLM, Ollama, etc.).

## 🚀 Features

- **Provider Agnostic**: Same interface works with any provider
- **Easy Provider Switching**: Change providers by just importing different classes
- **Tool Calling**: Automatic function introspection and execution
- **Streaming Support**: Both synchronous and streaming chat completions
- **Multimodal**: Support for text and image inputs
- **Type Safe**: Full type hints and Pydantic models
- **Extensible**: Easy to add new providers

## 📦 Installation

```bash
pip install -e .
```

## 🔧 Quick Start

### Basic Usage

```python
from unified_llm import OpenAILike

# Initialize provider with all parameters at once
provider = OpenAILike(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    # Connection settings
    base_url="http://localhost:8000/v1",
    api_key="your-api-key",
    # Common parameters
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    # Provider-specific parameters (vLLM/Ollama)
    top_k=40,
    repetition_penalty=1.1,
    min_p=0.05
)

# Basic chat - uses configured parameters
response = provider.chat([
    {"role": "user", "content": "Hello, world!"}
])
print(response.content)

# Clean up when done (or use context manager)
provider.close()

# Or use as context manager (recommended)
with OpenAILike(
    model_id="model-name", 
    base_url="http://localhost:8000/v1",
    temperature=0.8,
    top_k=50
) as provider:
    response = provider.chat([{"role": "user", "content": "Hello!"}])
    print(response.content)
```

### Parameter Configuration

Set all parameters once during initialization. The framework automatically separates common parameters from provider-specific ones:

```python
provider = OpenAILike(
    model_id="llama-3.1-8b",
    
    # Connection parameters
    base_url="http://localhost:8000/v1",
    api_key="your-key",
    timeout=30,
    
    # Common parameters (work across providers)
    temperature=0.7,        # Randomness
    max_tokens=1000,        # Response length
    top_p=0.9,             # Nucleus sampling
    frequency_penalty=0.0,  # Reduce repetition
    presence_penalty=0.0,   # Encourage new topics
    
    # Provider-specific parameters (vLLM/Ollama)
    top_k=40,              # Top-K sampling
    repetition_penalty=1.1, # Alternative repetition control
    min_p=0.05,            # Minimum probability threshold
    typical_p=1.0,         # Typical sampling
    mirostat=0,            # Mirostat sampling mode
    
    # Any other provider-specific parameters
    custom_param="value"
)

# All chat calls use these configured parameters
response = provider.chat(messages)
```

#### Parameter Categories

- **Common**: `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `seed`
- **vLLM/Ollama**: `top_k`, `repetition_penalty`, `min_p`, `typical_p`, `mirostat`, etc.
- **Connection**: `base_url`, `api_key`, `timeout`

### Tool Calling

```python
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: sunny, 22°C"

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

# Initialize with tools
provider = OpenAILike(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    tools=[get_weather, calculate],
    base_url="http://localhost:8000/v1"
)

# The framework automatically handles tool execution
response = provider.chat([
    {"role": "user", "content": "What's the weather in Paris and what's 15 * 23?"}
])
print(response.content)
```

### Streaming

```python
for chunk in provider.chat_stream([
    {"role": "user", "content": "Tell me a story"}
]):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
    if chunk.is_complete:
        break
```

### Reasoning Support

Enable reasoning content extraction to see the model's thinking process:

```python
# Standard response
response = provider.chat([
    {"role": "user", "content": "What's 2+2?"}
], enable_reasoning=False)
print(response.content)  # "4"

# With reasoning extraction
response = provider.chat([
    {"role": "user", "content": "What's 2+2?"}
], enable_reasoning=True)

if response.reasoning_content:
    print(f"Thinking: {response.reasoning_content}")
    
if response.reasoning_tokens:
    print(f"Reasoning tokens: {response.reasoning_tokens}")
    
print(f"Answer: {response.content}")
```

#### Streaming with Reasoning

```python
for chunk in provider.chat_stream([
    {"role": "user", "content": "Solve 2x + 5 = 15"}
], enable_reasoning=True):
    
    # Show reasoning process
    if chunk.reasoning_delta:
        print(chunk.reasoning_delta, end="", flush=True)
    
    # Indicate when reasoning is complete
    if chunk.is_reasoning_complete:
        print("\n--- Reasoning Complete ---")
    
    # Show final response
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
    
    if chunk.is_complete:
        break
```

### Metadata Access

Access provider-specific metadata for usage tracking, debugging, and analytics:

```python
response = provider.chat([
    {"role": "user", "content": "Hello!"}
])

# Access usage information
if 'usage' in response.metadata:
    usage = response.metadata['usage']
    print(f"Input tokens: {usage.get('prompt_tokens', 0)}")
    print(f"Output tokens: {usage.get('completion_tokens', 0)}")
    print(f"Total tokens: {usage.get('total_tokens', 0)}")

# Provider information
print(f"Provider: {response.metadata.get('provider')}")
print(f"Model: {response.metadata.get('model')}")
print(f"Finish reason: {response.metadata.get('finish_reason')}")

# Response ID for tracking
print(f"Response ID: {response.metadata.get('response_id')}")
```

#### Common Metadata Fields

- **`usage`**: Token usage statistics (`prompt_tokens`, `completion_tokens`, `total_tokens`)
- **`model`**: Actual model used for the response
- **`provider`**: Provider identifier (e.g., `"openai_like"`)
- **`endpoint`**: API endpoint used
- **`finish_reason`**: Why the response ended (`"stop"`, `"length"`, `"tool_calls"`, etc.)
- **`response_id`**: Unique identifier for the response
- **`created`**: Timestamp of response creation

### Multimodal

```python
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = provider.chat([
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "image_data": image_data}
        ]
    }
])
```

## 🌐 Supported Providers

### OpenAI-Compatible Endpoints

Works with any OpenAI-compatible API endpoint:

- **vLLM**: High-performance inference engine
- **Ollama**: Local LLM runner
- **Text Generation Inference**: Hugging Face's TGI
- **LM Studio**: Local inference server
- **Any OpenAI-compatible API**

#### vLLM Setup Example

```bash
# Install vLLM
pip install vllm

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000
```

#### Environment Variables

```bash
export OPENAI_LIKE_BASE_URL="http://localhost:8000/v1"
export OPENAI_LIKE_API_KEY="dummy-key"
```

## 🛠️ Configuration

### Direct Configuration

```python
provider = OpenAILike(
    model_id="model-name",
    base_url="http://localhost:8000/v1",
    api_key="your-key",
    temperature=0.7,
    max_tokens=1000,
    timeout=30
)
```

### Environment Variables

```bash
# Required
OPENAI_LIKE_BASE_URL=http://localhost:8000/v1
OPENAI_LIKE_API_KEY=your-api-key

# Optional
LLM_REQUEST_TIMEOUT=30
LLM_MAX_TOOL_CALLS=10
```

## 🔄 Provider Switching

One of the key benefits is easy provider switching:

```python
# Switch providers without changing your code
from unified_llm import OpenAILike as Provider
# from unified_llm import Bedrock as Provider  # Future provider

provider = Provider(model_id="model-name", tools=[my_tool])
response = provider.chat(messages)  # Same interface!
```

## 📚 Examples

Run the example script:

```bash
python example.py
```

This demonstrates:
- Basic chat functionality
- Tool calling with automatic execution
- Streaming responses
- Error handling

## 🏗️ Architecture

### Core Components

- **BaseProvider**: Abstract base class for all providers
- **OpenAILike**: OpenAI-compatible endpoint implementation
- **ChatResponse/ChatStreamResponse**: Unified response models
- **Tool Processing**: Automatic function introspection and schema generation

### Message Format

All providers use OpenAI-compatible message format:

```python
# Basic conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What's 2+2?"}
]

# Tool calling conversation (handled automatically by framework)
messages_with_tools = [
    {"role": "user", "content": "What's the weather in Paris?"},
    {
        "role": "assistant", 
        "content": "I'll check the weather for you.",
        "tool_calls": [{"id": "call_123", "name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}]
    },
    {"role": "tool", "content": "Weather in Paris: sunny, 22°C", "tool_call_id": "call_123"},
    {"role": "assistant", "content": "The weather in Paris is sunny and 22°C."}
]
```

## 🔧 Adding New Providers

Adding a new provider is straightforward:

```python
from unified_llm.base import BaseProvider
from unified_llm.models import ChatResponse, ChatStreamResponse

class NewProvider(BaseProvider):
    def _prepare_request(self, messages, stream=False):
        # Convert unified format to provider format
        return provider_request
    
    def _execute_request(self, request):
        # Make provider API call
        return provider_response
    
    def _parse_response(self, response):
        # Convert provider response to unified format
        return ChatResponse(content=text, thinking=thinking)
    
    def _construct_tools(self, functions):
        # Convert functions to provider tool schema
        return provider_tool_schema
```

## 🚧 Roadmap

- **AWS Bedrock Provider**: Direct Bedrock Converse API support
- **Anthropic Provider**: Direct Anthropic API support
- **Google Vertex AI Provider**: Vertex AI integration
- **Azure OpenAI Provider**: Azure-specific optimizations
- **Advanced Features**: Caching, retries, load balancing


