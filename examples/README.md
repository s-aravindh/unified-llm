# Unified LLM Interface - Examples

This directory contains comprehensive examples demonstrating the capabilities of the Unified LLM Interface. The examples are organized by provider type and showcase different aspects of the interface.

## Quick Start

1. **Run the interactive example menu:**
   ```bash
   cd examples/openai_like
   uv run python run_examples.py
   ```

2. **Run a specific example:**
   ```bash
   cd examples/openai_like  
   uv run python 01_basic_chat.py
   ```

3. **Run all examples:**
   ```bash
   cd examples/openai_like
   uv run python run_examples.py
   # Select option 11 (Run All Examples)
   ```

## Example Categories

### 🔧 Core Functionality
Essential features of the unified interface:

- **01_basic_chat.py** - Simple conversation without tools or streaming
- **02_tool_execution.py** - Single tool call execution with pure interface approach
- **03_multiple_tools.py** - Multiple tool calls in one response
- **04_reasoning.py** - Reasoning extraction from various formats

### ⚡ Streaming
Real-time response streaming capabilities:

- **05_basic_streaming.py** - Basic streaming without tools
- **06_streaming_with_tools.py** - Tool calls during streaming responses
- **07_streaming_reasoning.py** - Reasoning extraction during streaming
- **08_pattern_reasoning.py** - Tag-based reasoning pattern detection

### 💬 Conversation Management
Extended conversation and workflow patterns:

- **09_multi_turn.py** - Multi-turn conversations with context preservation
- **10_complex_workflow.py** - Advanced workflow orchestration and error handling

## Architecture Highlights

### Pure Interface Design
The examples demonstrate the **pure interface approach** where:
- **Providers return tool calls as data** (not auto-execute)
- **Applications control tool execution decisions**
- **ToolExecutor provides safe execution with error handling**
- **Clear separation of concerns**

```python
# Provider returns tool calls as data
response = provider.chat(messages)
if response.tool_calls:
    # Application decides whether to execute
    tool_results = executor.execute_all(response.tool_calls)
    # Application controls conversation continuation
```

### Provider-Specific Standardization
Each provider handles its own tool call format conversion:
- **OpenAI format** → **Standardized format** (internally)
- **Clean universal interface** for applications
- **Scalable architecture** for adding new providers

### Comprehensive Reasoning Support
Examples show reasoning extraction from multiple formats:
- **OpenAI o1 style reasoning tokens**
- **vLLM native reasoning fields** 
- **Pattern-based reasoning** (`<think>`, `<reasoning>` tags)
- **Streaming reasoning** with delta processing

## Configuration

### Default Configuration
Examples use local Ollama by default:
```python
{
    "model_id": "qwen3:4b",
    "base_url": "http://localhost:11434/v1", 
    "api_key": "fake-key",
    "temperature": 0.7,
    "max_tokens": 2000
}
```

### Override Configuration
Modify `common.py` or pass parameters to `get_default_provider()`:
```python
# Use OpenAI API
provider = get_default_provider(
    model_id="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key"
)

# Use vLLM server
provider = get_default_provider(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    base_url="http://localhost:8000/v1", 
    api_key="fake-key"
)
```

## Common Use Cases

### Basic Chat
```python
provider = get_default_provider()
messages = [{"role": "user", "content": "Hello!"}]
response = provider.chat(messages)
print(response.content)
```

### Tool Execution
```python
provider = get_default_provider(tools=[calculate])
executor = get_default_executor()

response = provider.chat(messages)
if response.tool_calls:
    tool_results = executor.execute_all(response.tool_calls)
    # Continue conversation with tool results
```

### Streaming
```python
for chunk in provider.stream(messages):
    if chunk.content_delta:
        print(chunk.content_delta, end="", flush=True)
    if chunk.tool_calls:
        # Handle tool calls during streaming
```

### Reasoning Extraction
```python
response = provider.chat(messages)
if response.reasoning:
    print(f"Reasoning ({response.reasoning_format}): {response.reasoning}")
```

## Tool Functions

Examples include three demo tools:

### calculate(expression: str) → float
Safe mathematical expression evaluation:
```python
result = calculate("25 * 4 + 10")  # Returns 110.0
```

### get_weather(location: str) → str
Mock weather information:
```python
weather = get_weather("Tokyo")  # Returns "Cloudy, 18°C"
```

### search_web(query: str) → str  
Mock web search results:
```python
results = search_web("AI news")  # Returns mock search results
```

## Error Handling

Examples demonstrate robust error handling:
- **Connection errors** with helpful hints
- **Tool execution errors** with recovery strategies
- **Malformed responses** with graceful degradation
- **Workflow interruption** with state preservation

## Performance Features

### Streaming Benefits
- **Immediate response start** (time to first token)
- **Real-time user feedback** during generation
- **Interruptible responses** for better UX
- **Progressive tool call detection**

### Tool Execution
- **Parallel execution** support via `execute_all()`
- **Selective execution** based on application policies
- **Validation before execution** with error reporting
- **Safe sandboxed execution** environment

## Extending Examples

### Adding New Examples
1. Create new `.py` file in `examples/openai_like/`
2. Follow naming convention: `NN_description.py`
3. Import from `common.py` for shared utilities
4. Add demo functions with `demo_` prefix
5. Update `EXAMPLES` dict in `run_examples.py`

### Adding New Tools
1. Define function with proper type hints
2. Add to `common.py` imports
3. Include in tool lists for relevant examples
4. Add to `get_default_executor()` if needed globally

### Adding New Providers
1. Create new provider subdirectory
2. Implement provider-specific examples
3. Create provider-specific `common.py`
4. Update main examples README

## Troubleshooting

### Common Issues

**"Connection refused" errors:**
- Ensure Ollama is running: `ollama serve`
- Check if model is available: `ollama list`
- Verify base_url in configuration

**"Model not found" errors:**
- Pull the model: `ollama pull qwen3:4b`
- Use available model from `ollama list`
- Check model name spelling

**Import errors:**
- Run from correct directory: `cd examples/openai_like`
- Use uv for dependencies: `uv run python script.py`
- Check if src/ directory is accessible

**Tool execution failures:**
- Check tool function signatures
- Verify JSON argument format
- Review ToolExecutor error messages

### Debug Mode
Enable verbose output by modifying `common.py`:
```python
def execute_tool_calls(executor, tool_calls, verbose=True):
    # Set verbose=True for detailed output
```

## Next Steps

After exploring the examples:

1. **Review the core interface** in `src/unified_llm/`
2. **Experiment with different providers** (OpenAI, vLLM, etc.)
3. **Build your own tools** following the examples
4. **Implement custom workflows** using the patterns shown
5. **Contribute new examples** for different use cases

For more information, see the main project documentation and API reference. 