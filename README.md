![Work In Progress](https://img.shields.io/badge/status-🚧%20WIP-orange)

# Unified LLM Interface

A **pure interface** library providing a consistent API across different Large Language Model providers, with optional tool execution capabilities.

## 🎯 Architecture Overview

### Pure Interface Design
- **Providers**: Handle format translation and return tool calls as **data only**
- **ToolExecutor**: Optional utility for safe tool execution with comprehensive error handling  
- **Applications**: Control all tool execution decisions and conversation management

### Tool Call Standardization
Each provider handles its own tool call standardization internally, converting from provider-native formats to a universal OpenAI-compatible format:

```python
# Standardized tool call format (used by all providers)
{
    "id": "call_abc123",
    "name": "function_name", 
    "arguments": "{\"param\": \"value\"}"
}
```

This ensures `ToolExecutor` works consistently with any provider without external standardization.

## 🧠 Reasoning Support

Comprehensive reasoning extraction across multiple formats:

### Supported Reasoning Formats

1. **OpenAI o1 Models**: Hidden reasoning with token counts
2. **vLLM**: Native `reasoning_content` field  
3. **Pattern-based**: `<think>`, `<reasoning>`, `<analysis>` tags
4. **Provider-specific**: `thinking` field support

### Reasoning Usage

```python
# Enable reasoning extraction
response = provider.chat(messages, enable_reasoning=True)

# Access reasoning content
if response.reasoning_content:
    print(f"Reasoning: {response.reasoning_content}")

# Access reasoning token count (OpenAI o1)  
if response.reasoning_tokens:
    print(f"Reasoning tokens: {response.reasoning_tokens}")

# Streaming with reasoning
for chunk in provider.chat_stream(messages, enable_reasoning=True):
    if chunk.reasoning_delta:
        print(f"Reasoning: {chunk.reasoning_delta}")
    
    if chunk.delta:
        print(chunk.delta, end="")
    
    if chunk.is_reasoning_complete:
        print("\n✅ Reasoning complete!")
```

### Reasoning Patterns Detected

```python
# These patterns are automatically extracted:
"<think>Step 1: Analyze the problem...</think>"
"<reasoning>First, I need to...</reasoning>" 
"<analysis>The key insight is...</analysis>"

# Provider-specific formats:
{"reasoning_content": "My thought process..."}  # vLLM
{"thinking": "Let me think..."}                 # Some providers
```

## 🚀 Quick Start

```python
from unified_llm import OpenAILike, ToolExecutor

# Define your tools
def calculate(expression: str) -> float:
    """Calculate a mathematical expression."""
    return eval(expression)  # Use proper math parser in production

def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny, 22°C in {location}"

# Initialize provider (handles standardization internally)
provider = OpenAILike(
    model_id="gpt-4o-mini",
    tools=[calculate, get_weather],  # For schema generation only
    temperature=0.7,
    api_key="your-api-key"
)

# Initialize optional tool executor  
executor = ToolExecutor(tools=[calculate, get_weather])

# Chat with pure interface
messages = [{"role": "user", "content": "What's 25 * 4? Also, weather in Tokyo?"}]
response = provider.chat(messages)

print(f"Assistant: {response.content}")

# Tool calls returned as data (provider handles standardization internally)
if response.tool_calls:
    print(f"Tool calls: {len(response.tool_calls)}")
    
    # Execute tools (application's choice)
    tool_results = []
    for tool_call in response.tool_calls:
        result = executor.execute_tool(tool_call)  # Works with any provider
        tool_results.append(result)
        print(f"Tool result: {result['content']}")
    
    # Continue conversation
    messages.extend([
        {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls},
        *tool_results
    ])
    
    final_response = provider.chat(messages)
    print(f"Final: {final_response.content}")
```

## 🔧 Provider-Specific Standardization

### OpenAI-Compatible Provider
Handles multiple OpenAI formats internally:

```python
# OpenAI Provider automatically standardizes these formats:

# 1. Full OpenAI format
{
    "id": "call_abc123",
    "type": "function",
    "function": {
        "name": "calculate",
        "arguments": "{\"expression\": \"25 * 4\"}"
    }
}

# 2. Simplified format (pass-through)
{
    "id": "call_abc123", 
    "name": "calculate",
    "arguments": "{\"expression\": \"25 * 4\"}"
}

# Both become standardized format internally
```

### Future Providers
Each new provider will implement its own `_standardize_tool_calls()` method:

```python
class BedrockProvider(BaseProvider):
    def _standardize_tool_calls(self, raw_tool_calls):
        """Convert Bedrock format to standardized format."""
        # Provider-specific standardization logic
        pass
```

## 📦 Installation

```bash
pip install -e .
```

## 🛠️ Configuration

### OpenAI-Compatible Endpoints

```python
# Official OpenAI API
provider = OpenAILike(
    model_id="gpt-4o-mini",
    api_key="sk-your-key",
    temperature=0.7
)

# Local vLLM server
provider = OpenAILike(
    model_id="llama-3.1-8b",
    base_url="http://localhost:8000/v1",
    api_key="fake",
    temperature=0.8,
    top_k=40  # vLLM-specific parameter
)

# Ollama with OpenAI compatibility
provider = OpenAILike(
    model_id="qwen3:4b",
    base_url="http://localhost:11434/v1", 
    api_key="fake",
    temperature=0.7
)
```

## 🔄 Streaming Support

```python
for chunk in provider.chat_stream(messages):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
    
    # Tool calls in streaming (standardized automatically)
    if chunk.tool_calls:
        print(f"\nTool calls: {chunk.tool_calls}")
    
    if chunk.is_complete:
        break
```

## 🛡️ Error Handling

```python
from unified_llm import ProviderError, ToolExecutionError

try:
    response = provider.chat(messages)
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            try:
                result = executor.execute_tool(tool_call)
            except ToolExecutionError as e:
                print(f"Tool execution failed: {e}")
                # Handle gracefully
                
except ProviderError as e:
    print(f"Provider error: {e}")
```

## 🧪 Testing

```bash
# Run tests
uv run python -m pytest tests/ -v

# Test specific functionality
uv run python -m pytest tests/test_openai_like.py -v
```

## 🏗️ Key Benefits

1. **Scalable Architecture**: Each provider handles its own standardization - no central bottleneck
2. **Provider Isolation**: Standardization logic stays with the provider that understands the format
3. **Consistent Interface**: All providers return identical tool call format
4. **Universal Compatibility**: ToolExecutor works with any provider automatically
5. **Maintainable**: Adding new providers doesn't affect existing code
6. **Comprehensive Reasoning**: Supports multiple reasoning formats across providers

## 📋 Roadmap

- [x] OpenAI-compatible provider with internal standardization
- [x] Pure interface design
- [x] Comprehensive tool execution
- [x] Multi-format reasoning support
- [ ] AWS Bedrock provider
- [ ] Anthropic Claude provider  
- [ ] Google Vertex AI provider
- [ ] Async support

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Add provider-specific standardization in the provider class
4. Add tests verifying standardization works correctly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🌟 Key Features

- **Multi-Provider Support**: OpenAI-compatible endpoints (vLLM, Ollama, etc.), with extensible architecture for AWS Bedrock and others
- **Pure Interface Design**: Tool calls returned as data, no automatic execution - applications control everything
- **Standardized Tool Format**: Consistent tool call structure across all providers for seamless ToolExecutor compatibility
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
- **Providers**: Format translation, API communication, schema generation, **tool call standardization**
- **ToolExecutor**: Optional safe tool execution with error handling
- **Applications**: Conversation management, tool execution decisions, orchestration

### 3. Constructor-Only Configuration
All parameters (common and provider-specific) are set once during initialization, eliminating runtime parameter confusion.

### 4. Standardized Tool Format
All providers return tool calls in a consistent format:
```python
{
    "id": "call_abc123",      # Unique identifier
    "name": "function_name",  # Function to call
    "arguments": "{...}"      # JSON string of arguments
}
```

## 📚 Advanced Usage

### Standardized Tool Call Format

All providers return tool calls in the same format regardless of their native format:

```python
# OpenAI native format (internal):
{
    "id": "call_abc123",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"Tokyo\"}"
    }
}

# Unified standardized format (what you get):
{
    "id": "call_abc123",
    "name": "get_weather", 
    "arguments": "{\"location\": \"Tokyo\"}"
}

# Works with ToolExecutor from any provider!
result = tool_executor.execute(tool_call)  # Always works
```

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
    
    # Tool calls in streaming (standardized format)
    if chunk.tool_calls:
        print(f"\nTool calls: {chunk.tool_calls}")
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
3. **Format Standardization**: Consistent tool call format across all providers
4. **Optional Execution**: ToolExecutor utility for safe execution when needed
5. **Application Control**: Full control over which tools to execute and when

```python
# Provider generates schemas, returns standardized tool calls as data
response = provider.chat(messages)
tool_calls = response.tool_calls  # Standardized format, never executed

# Application decides what to execute
if should_execute_tools(tool_calls):
    results = tool_executor.execute_all(tool_calls)  # Works with any provider
    messages.extend(results)
    final_response = provider.chat(messages)
```

### Tool Call Format Standardization

```python
# Different providers, same format for applications:

# vLLM/Ollama → Standardized
# OpenAI → Standardized  
# Bedrock → Standardized (future)
# All work with the same ToolExecutor!

for provider_type in ["vllm", "openai", "bedrock"]:
    provider = get_provider(provider_type)
    response = provider.chat(messages)
    
    # Same tool call format from all providers
    for tool_call in response.tool_calls:
        assert "id" in tool_call
        assert "name" in tool_call  
        assert "arguments" in tool_call
        
        # Same ToolExecutor works with all
        result = tool_executor.execute(tool_call)
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


