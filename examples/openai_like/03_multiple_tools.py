#!/usr/bin/env python3
"""
Example: Multiple Tools
Demonstrates: Multiple tool calls in a single response

This example shows:
- Provider with multiple tools available
- Single query triggering multiple tool calls
- Batch tool execution
- Complex conversation continuation
"""

from common import (
    get_default_provider, get_default_executor, 
    calculate, get_weather, search_web,
    print_section, handle_error, execute_tool_calls
)


def demo_multiple_tools():
    """Demonstrate multiple tool execution in one response."""
    
    print_section("Multiple Tools Example")
    
    try:
        # Initialize provider with multiple tools
        tools = [calculate, get_weather, search_web]
        provider = get_default_provider(tools=tools)
        executor = get_default_executor()
        
        print("🤖 Initializing provider with multiple tools...")
        print(f"   Model: {provider.model_id}")
        print(f"   Available tools: {[tool.__name__ for tool in tools]}")
        
        # Complex query that might trigger multiple tools
        messages = [
            {
                "role": "user", 
                "content": "I need help with several things: What's 150 * 8? What's the weather in Tokyo? And can you search for information about Python?"
            }
        ]
        
        print(f"\n👤 User: {messages[0]['content']}")
        print("🤖 Assistant analyzing request...")
        
        # Get response
        response = provider.chat(messages)
        print(f"🤖 Assistant: {response.content}")
        
        # Handle multiple tool calls
        if response.tool_calls:
            num_calls = len(response.tool_calls)
            print(f"\n📞 Multiple tool calls requested: {num_calls}")
            
            # Show all tool calls
            for i, tool_call in enumerate(response.tool_calls, 1):
                print(f"   {i}. {tool_call['name']}({tool_call['arguments']}) [ID: {tool_call['id']}]")
            
            # Execute all tools
            print(f"\n🔧 Executing {num_calls} tools...")
            tool_results = execute_tool_calls(executor, response.tool_calls)
            
            # Continue conversation
            messages.extend([
                {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls},
                *tool_results
            ])
            
            print(f"\n🔄 Getting final response with all tool results...")
            final_response = provider.chat(messages)
            print(f"🤖 Assistant: {final_response.content}")
            
        else:
            print("ℹ️ No tool calls were made")
        
        provider.close()
        
        print("\n✅ Multiple tools example completed successfully!")
        
    except Exception as e:
        handle_error(e, "multiple tools")


def demo_batch_vs_sequential():
    """Compare batch execution vs sequential execution."""
    
    print_section("Batch vs Sequential Tool Execution")
    
    try:
        provider = get_default_provider(tools=[calculate, get_weather])
        executor = get_default_executor()
        
        # Create a scenario with multiple calculations
        messages = [
            {
                "role": "user",
                "content": "Calculate 25 * 4, 100 / 5, and tell me the weather in London"
            }
        ]
        
        response = provider.chat(messages)
        
        if response.tool_calls:
            print(f"📊 Tool calls to execute: {len(response.tool_calls)}")
            
            # Method 1: Batch execution (using execute_tool_calls helper)
            print(f"\n🚀 Method 1: Batch Execution")
            import time
            start_time = time.time()
            
            batch_results = execute_tool_calls(executor, response.tool_calls, verbose=True)
            
            batch_time = time.time() - start_time
            print(f"   ⏱️ Batch execution time: {batch_time:.3f}s")
            
            # Method 2: Sequential execution with executor.execute_all
            print(f"\n🔄 Method 2: Using executor.execute_all()")
            start_time = time.time()
            
            sequential_results = executor.execute_all(response.tool_calls)
            
            sequential_time = time.time() - start_time
            print(f"   ⏱️ Sequential execution time: {sequential_time:.3f}s")
            
            # Show results are equivalent
            print(f"\n✅ Results comparison:")
            print(f"   Batch results count: {len(batch_results)}")
            print(f"   Sequential results count: {len(sequential_results)}")
            
            # Verify content matches
            for i, (batch_result, seq_result) in enumerate(zip(batch_results, sequential_results)):
                match = batch_result['content'] == seq_result['content']
                print(f"   Result {i+1} match: {'✅' if match else '❌'}")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "batch vs sequential")


def demo_selective_execution():
    """Demonstrate selective tool execution."""
    
    print_section("Selective Tool Execution")
    
    try:
        provider = get_default_provider(tools=[calculate, get_weather, search_web])
        executor = get_default_executor()
        
        messages = [
            {
                "role": "user",
                "content": "Calculate 50 * 3, get weather for Sydney, and search for news"
            }
        ]
        
        response = provider.chat(messages)
        
        if response.tool_calls:
            print(f"🎯 Selective execution example:")
            print(f"   Total tool calls: {len(response.tool_calls)}")
            
            # Application logic: only execute safe tools
            safe_tools = ['calculate', 'get_weather']  # Skip web search
            executed_results = []
            
            for tool_call in response.tool_calls:
                if tool_call['name'] in safe_tools:
                    print(f"   ✅ Executing safe tool: {tool_call['name']}")
                    result = executor.execute(tool_call)
                    executed_results.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call["id"]
                    })
                else:
                    print(f"   ❌ Skipping restricted tool: {tool_call['name']}")
                    executed_results.append({
                        "role": "tool",
                        "content": f"Tool '{tool_call['name']}' execution denied by application policy",
                        "tool_call_id": tool_call["id"]
                    })
            
            # Continue with partial results
            messages.extend([
                {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls},
                *executed_results
            ])
            
            final_response = provider.chat(messages)
            print(f"\n🤖 Final response: {final_response.content}")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "selective execution")


if __name__ == "__main__":
    demo_multiple_tools()
    demo_batch_vs_sequential()
    demo_selective_execution() 