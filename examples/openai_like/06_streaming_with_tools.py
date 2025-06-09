#!/usr/bin/env python3
"""
Example: Streaming with Tools
Demonstrates: Tool calls during streaming responses

This example shows:
- Streaming responses with tool call detection
- Tool calls collected during streaming
- Execution after stream completion
- Continuation of streaming after tool results
"""

import time
from common import (
    get_default_provider, get_default_executor, 
    calculate, get_weather, search_web,
    print_section, handle_error, execute_tool_calls
)


def demo_streaming_with_tools():
    """Demonstrate tool calls during streaming."""
    
    print_section("Streaming with Tools Example")
    
    try:
        provider = get_default_provider(tools=[calculate, get_weather])
        executor = get_default_executor()
        
        print("🤖 Initializing streaming provider with tools...")
        print(f"   Model: {provider.model_id}")
        print(f"   Available tools: {[tool.__name__ for tool in provider.tools]}")
        
        messages = [
            {"role": "user", "content": "What's 25 * 8? Also, what's the weather like in Tokyo?"}
        ]
        
        print(f"\n👤 User: {messages[0]['content']}")
        print("🤖 Assistant: ", end="", flush=True)
        
        # Stream and collect tool calls
        full_content = ""
        accumulated_tool_calls = []
        
        for chunk in provider.stream(messages):
            # Handle content delta
            if chunk.content_delta:
                print(chunk.content_delta, end="", flush=True)
                full_content += chunk.content_delta
            
            # Collect tool calls as they come in
            if chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    # Check if this tool call is already in our list
                    existing_call = next(
                        (tc for tc in accumulated_tool_calls if tc.get('id') == tool_call.get('id')), 
                        None
                    )
                    
                    if existing_call:
                        # Update existing call with new information
                        existing_call.update(tool_call)
                    else:
                        # Add new tool call
                        accumulated_tool_calls.append(tool_call.copy())
            
            if chunk.finish_reason:
                break
        
        print(f"\n")
        
        # Execute collected tool calls
        if accumulated_tool_calls:
            print(f"\n📞 Tool calls collected during streaming: {len(accumulated_tool_calls)}")
            
            for i, tool_call in enumerate(accumulated_tool_calls, 1):
                print(f"   {i}. {tool_call['name']}({tool_call.get('arguments', 'N/A')}) [ID: {tool_call.get('id', 'N/A')}]")
            
            # Execute tools
            print(f"\n🔧 Executing tools...")
            tool_results = execute_tool_calls(executor, accumulated_tool_calls)
            
            # Continue conversation with tool results
            messages.extend([
                {"role": "assistant", "content": full_content, "tool_calls": accumulated_tool_calls},
                *tool_results
            ])
            
            print(f"\n🔄 Continuing stream with tool results...")
            print("🤖 Assistant: ", end="", flush=True)
            
            for chunk in provider.stream(messages):
                if chunk.content_delta:
                    print(chunk.content_delta, end="", flush=True)
                
                if chunk.finish_reason:
                    break
        
        print(f"\n")
        provider.close()
        
        print("\n✅ Streaming with tools example completed!")
        
    except Exception as e:
        handle_error(e, "streaming with tools")


def demo_progressive_tool_calls():
    """Demonstrate progressive tool call collection during streaming."""
    
    print_section("Progressive Tool Call Collection")
    
    try:
        provider = get_default_provider(tools=[calculate, get_weather, search_web])
        executor = get_default_executor()
        
        messages = [
            {"role": "user", "content": "Calculate 15 * 12, check weather in London, and search for AI news"}
        ]
        
        print(f"👤 User: {messages[0]['content']}")
        print("🤖 Assistant thinking and planning...")
        
        # Track tool call building process
        tool_call_states = {}
        complete_tool_calls = []
        
        for chunk in provider.stream(messages):
            if chunk.content_delta:
                print(f"📝 Content: {chunk.content_delta.strip()}")
            
            if chunk.tool_calls:
                print(f"🔧 Tool call chunk received: {len(chunk.tool_calls)} items")
                
                for tool_call in chunk.tool_calls:
                    call_id = tool_call.get('id', 'unknown')
                    
                    # Initialize or update tool call state
                    if call_id not in tool_call_states:
                        tool_call_states[call_id] = {}
                        print(f"   🆕 New tool call started: {call_id}")
                    
                    # Update state
                    tool_call_states[call_id].update(tool_call)
                    
                    # Check if tool call is complete
                    current_state = tool_call_states[call_id]
                    if all(key in current_state for key in ['id', 'name', 'arguments']):
                        if call_id not in [tc['id'] for tc in complete_tool_calls]:
                            complete_tool_calls.append(current_state.copy())
                            print(f"   ✅ Tool call completed: {current_state['name']}")
            
            if chunk.finish_reason:
                print(f"🏁 Stream finished: {chunk.finish_reason}")
                break
        
        # Show final state
        print(f"\n📊 Final Tool Call Analysis:")
        print(f"   Total tool calls started: {len(tool_call_states)}")
        print(f"   Complete tool calls: {len(complete_tool_calls)}")
        
        for tool_call in complete_tool_calls:
            print(f"   - {tool_call['name']}: {tool_call.get('arguments', 'N/A')}")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "progressive tool calls")


def demo_streaming_tool_timing():
    """Demonstrate timing analysis of streaming with tools."""
    
    print_section("Streaming Tool Timing Analysis")
    
    try:
        provider = get_default_provider(tools=[calculate])
        executor = get_default_executor()
        
        messages = [
            {"role": "user", "content": "Calculate 234 * 567 and explain the steps"}
        ]
        
        print(f"👤 User: {messages[0]['content']}")
        
        # Track timing
        start_time = time.time()
        first_content_time = None
        first_tool_call_time = None
        stream_end_time = None
        
        accumulated_tool_calls = []
        
        print("🤖 Assistant: ", end="", flush=True)
        
        for chunk in provider.stream(messages):
            current_time = time.time() - start_time
            
            if chunk.content_delta and first_content_time is None:
                first_content_time = current_time
                print(f"⏱️  First content at {current_time:.2f}s")
                print("🤖 Content: ", end="", flush=True)
            
            if chunk.content_delta:
                print(chunk.content_delta, end="", flush=True)
            
            if chunk.tool_calls and first_tool_call_time is None:
                first_tool_call_time = current_time
                print(f"\n⏱️  First tool call at {current_time:.2f}s")
            
            if chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    existing_call = next(
                        (tc for tc in accumulated_tool_calls if tc.get('id') == tool_call.get('id')), 
                        None
                    )
                    if existing_call:
                        existing_call.update(tool_call)
                    else:
                        accumulated_tool_calls.append(tool_call.copy())
            
            if chunk.finish_reason:
                stream_end_time = current_time
                break
        
        print(f"\n")
        
        # Execute tools
        if accumulated_tool_calls:
            tool_exec_start = time.time() - start_time
            tool_results = execute_tool_calls(executor, accumulated_tool_calls, verbose=False)
            tool_exec_end = time.time() - start_time
            
            print(f"⏱️  Tool execution: {tool_exec_end - tool_exec_start:.2f}s")
        
        # Show timing summary
        print(f"\n📊 Timing Analysis:")
        if first_content_time:
            print(f"   Time to first content: {first_content_time:.2f}s")
        if first_tool_call_time:
            print(f"   Time to first tool call: {first_tool_call_time:.2f}s")
        if stream_end_time:
            print(f"   Total streaming time: {stream_end_time:.2f}s")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "streaming tool timing")


if __name__ == "__main__":
    demo_streaming_with_tools()
    demo_progressive_tool_calls()
    demo_streaming_tool_timing() 