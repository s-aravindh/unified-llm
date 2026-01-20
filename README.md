> use litellm it has better support and better api

# Unified LLM Interface

A **pure interface** library providing a consistent API across different Large Language Model providers, with optional tool execution capabilities.

## 🎯 Architecture Overview

### Pure Interface Design
- **Providers**: Handle format translation and return tool calls as **data only**
- **ToolExecutor**: Optional utility for safe tool execution with comprehensive error handling  
- **Applications**: Control all tool execution decisions and conversation management

### Provider Structure
```
unified_llm/
├── providers/
│   ├── openai_like/     # OpenAI-compatible endpoints (OpenAI, vLLM, Ollama)
│   └── bedrock/         # AWS Bedrock foundation models
├── base.py              # Abstract base provider
├── models.py            # Response data models
├── tool_executor.py     # Optional tool execution utility
└── exceptions.py        # Error handling
```

## 📦 Installation

```bash
pip install -e .
```

## 🚀 Quick Start

### Import Providers

```python
# Main package imports (recommended)
from unified_llm import OpenAILike, Bedrock, ToolExecutor

# Or specific provider imports
from unified_llm.providers.openai_like import OpenAILike
from unified_llm.providers.bedrock import Bedrock
```

## 🗣️ Basic Chat Examples

### OpenAI-Compatible Provider

```python
from unified_llm import OpenAILike

# Initialize provider
provider = OpenAILike(
    model_id="gpt-4o-mini",
    api_key="your-openai-key",
    temperature=0.7,
    max_tokens=1000
)

# Basic chat
messages = [
    {"role": "user", "content": "Explain quantum computing in simple terms"}
]

response = provider.chat(messages)
print(f"Assistant: {response.content}")
print(f"Usage: {response.metadata['usage']}")
```

### AWS Bedrock Provider

```python
from unified_llm import Bedrock

# Initialize with AWS credentials
provider = Bedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name="us-east-1",
    # AWS credentials will be resolved automatically from:
    # 1. AWS profile, 2. Environment variables, 3. IAM roles
)

# Basic chat
messages = [
    {"role": "user", "content": "Write a haiku about programming"}
]

response = provider.chat(messages)
print(f"Claude: {response.content}")
```

### Local vLLM Server

```python
provider = OpenAILike(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="fake",  # vLLM doesn't require real API key
    temperature=0.8,
    top_k=40  # vLLM-specific parameter
)

response = provider.chat(messages)
print(response.content)
```

## 🔄 Streaming Chat

### Basic Streaming

```python
# OpenAI streaming
messages = [{"role": "user", "content": "Tell me a story about AI"}]

print("Assistant: ", end="")
for chunk in provider.chat_stream(messages):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
    
    if chunk.is_complete:
        print("\n✅ Stream complete")
        break
```

### Bedrock Streaming

```python
# AWS Bedrock streaming
for chunk in bedrock_provider.chat_stream(messages):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
    
    # Access streaming metadata
    if chunk.metadata.get('usage'):
        print(f"\nTokens used: {chunk.metadata['usage']}")
```

## 🛠️ Tool Use Examples

### Define Tools

```python
def calculate(expression: str) -> float:
    """Calculate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
    """
    # Use a proper math parser in production
    return eval(expression)

def get_weather(location: str) -> str:
    """Get current weather for a location.
    
    Args:
        location: City name to get weather for
        
    Returns:
        Weather description
    """
    # Simulate weather API call
    return f"Weather in {location}: Sunny, 22°C"

def search_web(query: str) -> str:
    """Search the web for information.
    
    Args:
        query: Search query
        
    Returns:
        Search results summary
    """
    return f"Search results for '{query}': [Simulated results]"
```

### Tool Use with OpenAI

```python
from unified_llm import OpenAILike, ToolExecutor

# Initialize provider with tools
provider = OpenAILike(
    model_id="gpt-4o-mini",
    tools=[calculate, get_weather, search_web],
    api_key="your-key"
)

# Initialize tool executor
executor = ToolExecutor(tools=[calculate, get_weather, search_web])

# Chat with tool capabilities
messages = [
    {"role": "user", "content": "What's 15 * 23? Also get weather in Tokyo and search for 'latest AI news'"}
]

response = provider.chat(messages)
print(f"Assistant: {response.content}")

# Execute tools if requested
if response.tool_calls:
    print(f"\n🔧 Executing {len(response.tool_calls)} tools...")
    
    # Execute all tools
    tool_results = executor.execute_all(response.tool_calls)
    
    for result in tool_results:
        print(f"Tool result: {result['content']}")
    
    # Continue conversation with tool results
    messages.extend([
        {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls},
        *tool_results
    ])
    
    final_response = provider.chat(messages)
    print(f"\nFinal answer: {final_response.content}")
```

### Tool Use with Bedrock

```python
from unified_llm import Bedrock, ToolExecutor

# Initialize Bedrock with tools
provider = Bedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    tools=[calculate, get_weather],
    region_name="us-east-1"
)

executor = ToolExecutor(tools=[calculate, get_weather])

# Same tool workflow as OpenAI
messages = [{"role": "user", "content": "Calculate 45 * 67 and get weather in London"}]

response = provider.chat(messages)
if response.tool_calls:
    tool_results = executor.execute_all(response.tool_calls)
    # Continue conversation...
```

### Streaming Tool Use

```python
# Tools can be called during streaming
messages = [{"role": "user", "content": "Calculate 123 * 456 and explain the result"}]

current_tools = []
content_parts = []

for chunk in provider.chat_stream(messages):
    # Collect content
    if chunk.delta:
        content_parts.append(chunk.delta)
        print(chunk.delta, end="", flush=True)
    
    # Collect tool calls
    if chunk.tool_calls:
        current_tools.extend(chunk.tool_calls)
        print(f"\n🔧 Tool call: {chunk.tool_calls}")
    
    if chunk.is_complete:
        print("\n✅ Stream complete")
        
        # Execute tools if any were called
        if current_tools:
            tool_results = executor.execute_all(current_tools)
            print("Tool results:", tool_results)
        break
```

## 💬 Multi-Turn Conversations

### Basic Multi-Turn

```python
from unified_llm import OpenAILike

provider = OpenAILike(model_id="gpt-4o-mini", api_key="your-key")

# Initialize conversation
conversation = [
    {"role": "system", "content": "You are a helpful programming assistant."}
]

def chat_turn(user_message):
    conversation.append({"role": "user", "content": user_message})
    
    response = provider.chat(conversation)
    conversation.append({"role": "assistant", "content": response.content})
    
    return response.content

# Multi-turn conversation
print("🤖 Assistant: Hello! I'm ready to help with programming questions.")

response1 = chat_turn("How do I create a Python list?")
print(f"🤖 Assistant: {response1}")

response2 = chat_turn("Now show me how to add items to it")
print(f"🤖 Assistant: {response2}")

response3 = chat_turn("What about removing items?")
print(f"🤖 Assistant: {response3}")
```

### Multi-Turn with Tools

```python
from unified_llm import Bedrock, ToolExecutor

def save_note(content: str, filename: str = "note.txt") -> str:
    """Save a note to a file."""
    # Simulate saving
    return f"Note saved to {filename}: {content[:50]}..."

def list_files() -> str:
    """List all saved files."""
    return "Files: note.txt, todo.txt, ideas.txt"

# Initialize
provider = Bedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    tools=[save_note, list_files]
)
executor = ToolExecutor(tools=[save_note, list_files])

conversation = []

def process_turn(user_input):
    conversation.append({"role": "user", "content": user_input})
    
    response = provider.chat(conversation)
    
    # Handle tool calls
    if response.tool_calls:
        # Add assistant message with tools
        conversation.append({
            "role": "assistant", 
            "content": response.content,
            "tool_calls": response.tool_calls
        })
        
        # Execute tools and add results
        tool_results = executor.execute_all(response.tool_calls)
        conversation.extend(tool_results)
        
        # Get final response
        final_response = provider.chat(conversation)
        conversation.append({"role": "assistant", "content": final_response.content})
        
        return final_response.content
    else:
        conversation.append({"role": "assistant", "content": response.content})
        return response.content

# Multi-turn with tools
response1 = process_turn("Save a note: 'Meeting with team tomorrow at 3pm'")
print(f"Assistant: {response1}")

response2 = process_turn("What files do I have?")
print(f"Assistant: {response2}")

response3 = process_turn("Save another note about buying groceries")
print(f"Assistant: {response3}")
```

### Streaming Multi-Turn

```python
conversation = []

def streaming_turn(user_input):
    conversation.append({"role": "user", "content": user_input})
    
    print("🤖 Assistant: ", end="")
    full_content = ""
    tools_called = []
    
    for chunk in provider.chat_stream(conversation):
        if chunk.delta:
            full_content += chunk.delta
            print(chunk.delta, end="", flush=True)
        
        if chunk.tool_calls:
            tools_called.extend(chunk.tool_calls)
        
        if chunk.is_complete:
            print()  # New line
            
            # Add assistant message
            assistant_msg = {"role": "assistant", "content": full_content}
            if tools_called:
                assistant_msg["tool_calls"] = tools_called
            conversation.append(assistant_msg)
            
            # Execute tools if needed
            if tools_called:
                tool_results = executor.execute_all(tools_called)
                conversation.extend(tool_results)
                
                # Get final response
                final_response = provider.chat(conversation)
                conversation.append({"role": "assistant", "content": final_response.content})
                print(f"🔧 After tools: {final_response.content}")
            
            break

# Streaming multi-turn
streaming_turn("Hello, can you help me with file management?")
streaming_turn("Save a note with today's tasks")
streaming_turn("Show me all my files")
```

## 🧠 Reasoning Support

### Enable Reasoning Extraction

```python
# Works with reasoning-capable models (Claude 3.7+, OpenAI o1, etc.)
response = provider.chat(messages, enable_reasoning=True)

# Access reasoning content
if response.reasoning_content:
    print(f"🧠 Reasoning: {response.reasoning_content}")

# Access reasoning token count (if provided)
if response.reasoning_tokens:
    print(f"🔢 Reasoning tokens: {response.reasoning_tokens}")
```

### Streaming with Reasoning

```python
for chunk in provider.chat_stream(messages, enable_reasoning=True):
    # Reasoning content (separate from final response)
    if chunk.reasoning_delta:
        print(f"💭 {chunk.reasoning_delta}", end="")
    
    # Final response content
    if chunk.delta:
        print(f"🤖 {chunk.delta}", end="")
    
    # Reasoning phase completion
    if chunk.is_reasoning_complete:
        print("\n✅ Reasoning complete")
    
    # Full completion
    if chunk.is_complete:
        print("\n✅ Response complete")
        break
```

## 🔧 Advanced Configuration

### Provider-Specific Parameters

```python
# OpenAI with custom parameters
openai_provider = OpenAILike(
    model_id="gpt-4o",
    api_key="your-key",
    temperature=0.7,
    max_tokens=2000,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    stop=["END", "STOP"]
)

# Bedrock with model-specific parameters
bedrock_provider = Bedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name="us-west-2",
    temperature=0.8,
    max_tokens=1500,
    top_p=0.95,
    top_k=40,  # Anthropic-specific
    reasoning_budget_tokens=2000  # For reasoning models
)

# vLLM with advanced parameters
vllm_provider = OpenAILike(
    model_id="meta-llama/Llama-3.1-70B-Instruct",
    base_url="http://localhost:8000/v1",
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,  # vLLM-specific
    min_p=0.05,              # vLLM-specific
    typical_p=1.0            # vLLM-specific
)
```

### Multiple Providers

```python
# Use different providers for different tasks
providers = {
    "creative": OpenAILike(model_id="gpt-4o", temperature=0.9),
    "analytical": Bedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", temperature=0.3),
    "local": OpenAILike(model_id="llama-3.1-8b", base_url="http://localhost:8000/v1")
}

def chat_with_provider(provider_name, message):
    provider = providers[provider_name]
    response = provider.chat([{"role": "user", "content": message}])
    return response.content

# Route different queries to appropriate providers
creative_response = chat_with_provider("creative", "Write a poem about coding")
analytical_response = chat_with_provider("analytical", "Analyze this data pattern")
local_response = chat_with_provider("local", "Simple math: 2+2")
```

## 🛡️ Error Handling

```python
from unified_llm import ProviderError, ToolExecutionError, ConfigurationError

try:
    response = provider.chat(messages)
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            try:
                result = executor.execute(tool_call)
                print(f"Tool result: {result}")
            except ToolExecutionError as e:
                print(f"Tool execution failed: {e}")
                # Handle gracefully with error message
                
except ProviderError as e:
    print(f"Provider API error: {e}")
    print(f"Status code: {e.status_code}")
    
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 🧪 Testing

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Test specific provider
uv run python -m pytest tests/test_providers/test_openai_like.py -v
uv run python -m pytest tests/test_providers/test_bedrock.py -v

# Test streaming
uv run python -m pytest tests/ -k "stream" -v
```

## 🏗️ Key Benefits

- **🔧 Pure Interface**: Tool calls returned as data, applications control execution
- **🎯 Standardized**: All providers return identical tool call format
- **🚀 Provider Agnostic**: Switch between OpenAI, Bedrock, vLLM seamlessly  
- **🛠️ Flexible**: Each provider handles its own format conversion internally
- **⚡ Streaming**: Full streaming support with proper state management
- **🧠 Reasoning**: Extract reasoning content from compatible models
- **🔒 Type Safe**: Full type annotations and mypy support
- **📦 Extensible**: Easy to add new providers following the same pattern

## 📚 Examples Directory

Check the `examples/` directory for complete working examples:
- `examples/basic_chat.py` - Simple chat examples
- `examples/tool_use.py` - Tool calling examples  
- `examples/streaming.py` - Streaming examples
- `examples/multi_turn.py` - Conversation examples
- `examples/reasoning.py` - Reasoning extraction examples



**readme generated with claude 4, use with caution**
