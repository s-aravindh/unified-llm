![Work In Progress](https://img.shields.io/badge/status-🚧%20WIP-orange)

# Unified LLM Interface


A unified interface for Large Language Models (LLMs) supporting multiple providers, with tool integration, reasoning extraction, and flexible configuration. **Pure interface design**: providers return tool calls as structured data without automatic execution - applications maintain full control over tool execution and conversation flow.

## 🌟 Key Features

- **Multi-Provider Support**: OpenAI-compatible endpoints (vLLM, Ollama, etc.), with extensible architecture for AWS Bedrock and others
- **Pure Interface Design**: Tool calls returned as data, no automatic execution - applications control everything
- **Comprehensive Tool Integration**: Optional ToolExecutor utility for safe tool execution with error handling
- **Advanced Reasoning Support**: Extract reasoning content from provider responses (e.g., OpenAI o1, pattern-based extraction)
- **Flexible Configuration**: Constructor-only parameters with automatic common/provider-specific separation
- **Streaming Support**: Real-time response streaming with tool call support
- **Rich Metadata**: Extensible metadata system for provider-specific information
- **Modern HTTP Client**: Built on httpx with connection management and context manager support

## 🏗️ Architecture Principles

### 1. Pure Interface Design
Providers are **pure interfaces** that translate between formats and return structured data. They do **NOT**:
- Execute tools automatically
- Manage conversation state
- Make tool execution decisions

### 2. Separation of Concerns
- **Providers**: Format translation, API communication, schema generation
- **ToolExecutor**: Optional safe tool execution with error handling
- **Applications**: Conversation management, tool execution decisions, orchestration

### 3. Constructor-Only Configuration
All parameters (common and provider-specific) are set once during initialization, eliminating runtime parameter confusion.

## 🚀 Quick Start

### Basic Installation

```bash
pip install unified-llm
```

### Development Installation

```bash
git clone <repository-url>
cd unified-llm
uv sync --dev
```

### Basic Usage

```python
from unified_llm import OpenAILike, ToolExecutor

# Define tools
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: sunny, 22°C"

def calculate(expression: str) -> float:
    """Calculate mathematical expression."""
    return eval(expression)

# Initialize provider (pure interface)
provider = OpenAILike(
    model_id="qwen3:4b",
    base_url="http://localhost:11434/v1",
    api_key="dummy-key",
    tools=[get_weather, calculate],  # For schema generation only
    temperature=0.7,
    max_tokens=1000
)

# Initialize tool executor (optional utility)
tool_executor = ToolExecutor(tools=[get_weather, calculate])

# Chat with pure interface
messages = [
    {"role": "user", "content": "What's the weather in Tokyo and what's 5*7?"}
]

response = provider.chat(messages)
print(f"Assistant: {response.content}")

# Application controls tool execution
if response.tool_calls:
    print(f"Tool calls requested: {len(response.tool_calls)}")
    
    # Add assistant message
    messages.append({
        "role": "assistant",
        "content": response.content,
        "tool_calls": response.tool_calls
    })
    
    # Execute tools (application decision)
    for tool_call in response.tool_calls:
        result = tool_executor.execute(tool_call)
        print(f"Executed {tool_call['name']}: {result}")
        
        # Add tool result
        messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call["id"]
        })
    
    # Continue conversation
    final_response = provider.chat(messages)
    print(f"Final response: {final_response.content}")
```

## 📚 Advanced Usage

### Tool Call Validation

```python
# Validate tool calls before execution
errors = tool_executor.validate_tool_call(tool_call)
if errors:
    print(f"Validation errors: {errors}")
else:
    result = tool_executor.execute(tool_call)
```

### Selective Tool Execution

```python
for tool_call in response.tool_calls:
    # Application logic for tool safety
    if tool_call['name'] in ['safe_tool1', 'safe_tool2']:
        result = tool_executor.execute(tool_call)
        # Add to conversation...
    else:
        # Reject unsafe tools
        messages.append({
            "role": "tool",
            "content": f"Tool '{tool_call['name']}' execution denied",
            "tool_call_id": tool_call["id"]
        })
```

### Batch Tool Execution

```python
# Execute multiple tools at once
tool_results = tool_executor.execute_all(response.tool_calls)

# Add all results to conversation
messages.extend(tool_results)
```

### Reasoning Support

```python
response = provider.chat(messages, enable_reasoning=True)

if response.reasoning_content:
    print(f"Reasoning: {response.reasoning_content}")
    print(f"Reasoning tokens: {response.reasoning_tokens}")

print(f"Final answer: {response.content}")
```

### Streaming with Tool Calls

```python
for chunk in provider.chat_stream(messages):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
    
    # Tool calls in streaming (final chunk)
    if chunk.tool_calls:
        # Handle tool calls...
        pass
    
    if chunk.is_complete:
        break
```

### Flexible Configuration

```python
# All parameters set during initialization
provider = OpenAILike(
    model_id="qwen3:4b",
    base_url="http://localhost:11434/v1",
    # Common parameters
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    # Provider-specific parameters
    top_k=40,
    repetition_penalty=1.1,
    min_p=0.05,
    # Custom parameters
    custom_provider_param="value"
)

# Parameters are automatically separated
print(provider.common_params)    # Common parameters
print(provider.provider_params)  # Provider-specific parameters
```

### Context Management

```python
# Automatic resource cleanup
with OpenAILike(model_id="model", **config) as provider:
    response = provider.chat(messages)
    # Connection automatically closed
```

### Metadata Access

```python
response = provider.chat(messages)

# Access provider-specific metadata
if response.metadata:
    usage = response.metadata.get('usage', {})
    print(f"Tokens used: {usage}")
    
    model_info = response.metadata.get('model')
    finish_reason = response.metadata.get('finish_reason')
```

## 🔧 Tool Integration

The interface supports flexible tool integration through:

1. **Schema Generation**: Tools provided to providers for OpenAI-compatible schema generation
2. **Pure Interface**: Tool calls returned as structured data, not executed
3. **Optional Execution**: ToolExecutor utility for safe execution when needed
4. **Application Control**: Full control over which tools to execute and when

```python
# Provider generates schemas, returns tool calls as data
response = provider.chat(messages)
tool_calls = response.tool_calls  # Data only, not executed

# Application decides what to execute
if should_execute_tools(tool_calls):
    results = tool_executor.execute_all(tool_calls)
    messages.extend(results)
    final_response = provider.chat(messages)
```

## 🧠 Reasoning Support

Advanced reasoning extraction supports multiple formats:

- **OpenAI o1 Style**: Separate reasoning/content with token counts
- **Pattern-based**: Extract `<thinking>`, `<reasoning>` tags from content  
- **Provider Native**: vLLM/Ollama native reasoning_content fields
- **Streaming**: Real-time reasoning and content separation

```python
# Enable reasoning extraction
response = provider.chat(messages, enable_reasoning=True)

# Access reasoning information
reasoning = response.reasoning_content    # Extracted reasoning
content = response.content               # Final answer
tokens = response.reasoning_tokens       # Token usage for reasoning

# Streaming reasoning
for chunk in provider.chat_stream(messages, enable_reasoning=True):
    if chunk.reasoning_delta:
        print(f"Reasoning: {chunk.reasoning_delta}")
    if chunk.delta:
        print(f"Response: {chunk.delta}")
```

## 🏃‍♂️ Running Examples

```bash
# Basic example
uv run example.py

# Set custom endpoint
OPENAI_LIKE_BASE_URL=http://localhost:11434/v1 uv run example.py

# With API key
OPENAI_LIKE_API_KEY=your-key-here uv run example.py
```

## 🔧 Supported Providers

### OpenAI-Compatible
- **vLLM**: High-performance inference server
- **Ollama**: Local model serving
- **OpenAI**: Direct OpenAI API
- **Any OpenAI-compatible endpoint**

### Configuration Examples

```python
# vLLM
provider = OpenAILike(
    model_id="microsoft/DialoGPT-medium",
    base_url="http://localhost:8000/v1",
    temperature=0.7,
    max_tokens=1000,
    # vLLM-specific
    top_k=40,
    repetition_penalty=1.1
)

# Ollama
provider = OpenAILike(
    model_id="qwen3:4b",
    base_url="http://localhost:11434/v1",
    api_key="dummy-key",
    temperature=0.7,
    # Ollama-specific
    top_k=40,
    min_p=0.05
)

# OpenAI
provider = OpenAILike(
    model_id="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=1000
)
```

## 🔮 Roadmap

- [ ] AWS Bedrock provider
- [ ] Google Vertex AI provider
- [ ] Anthropic Claude provider
- [ ] Enhanced multimodal support
- [ ] Performance benchmarking
- [ ] Advanced tool orchestration patterns


