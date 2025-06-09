#!/usr/bin/env python3
"""
Example: Tool Execution
Demonstrates: Single tool call execution with the pure interface approach

This example shows:
- Provider returns tool calls as data (pure interface)
- Application controls tool execution decisions
- ToolExecutor provides safe execution with error handling
- Conversation continuation with tool results
"""

from common import (
    get_default_provider, get_default_executor, 
    calculate, print_section, handle_error, execute_tool_calls
)


def demo_tool_execution():
    """Demonstrate single tool execution."""
    
    print_section("Tool Execution Example")
    
    try:
        # Initialize provider with tools (for schema generation only)
        provider = get_default_provider(tools=[calculate])
        executor = get_default_executor()
        
        print("🤖 Initializing provider with tools...")
        print(f"   Model: {provider.model_id}")
        print(f"   Available tools: {[tool.__name__ for tool in provider.tools]}")
        
        # Ask for calculation
        messages = [
            {"role": "user", "content": "What is 25 * 4 + 10? Please calculate this for me."}
        ]
        
        print(f"\n👤 User: {messages[0]['content']}")
        print("🤖 Assistant thinking...")
        
        # Get response - tool calls returned as data
        response = provider.chat(messages)
        print(f"🤖 Assistant: {response.content}")
        
        # Check if tools were requested
        if response.tool_calls:
            print(f"\n📞 Tool calls requested: {len(response.tool_calls)}")
            
            # Show tool calls in detail
            for i, tool_call in enumerate(response.tool_calls, 1):
                print(f"   {i}. Function: {tool_call['name']}")
                print(f"      Arguments: {tool_call['arguments']}")
                print(f"      ID: {tool_call['id']}")
            
            # Application decides to execute tools
            print(f"\n🔧 Executing tools...")
            tool_results = execute_tool_calls(executor, response.tool_calls)
            
            # Continue conversation with tool results
            messages.extend([
                {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls},
                *tool_results
            ])
            
            print(f"\n🔄 Continuing conversation with tool results...")
            final_response = provider.chat(messages)
            print(f"🤖 Assistant: {final_response.content}")
            
        else:
            print("ℹ️ No tool calls were made in this response")
        
        provider.close()
        
        print("\n✅ Tool execution example completed successfully!")
        
    except Exception as e:
        handle_error(e, "tool execution")


def demo_tool_call_details():
    """Show detailed information about tool calls."""
    
    print_section("Tool Call Details")
    
    try:
        provider = get_default_provider(tools=[calculate])
        executor = get_default_executor()
        
        messages = [
            {"role": "user", "content": "Calculate (15 + 25) * 2"}
        ]
        
        response = provider.chat(messages)
        
        if response.tool_calls:
            print("🔍 Detailed tool call analysis:")
            
            for tool_call in response.tool_calls:
                print(f"\n📋 Tool Call Details:")
                print(f"   📝 Raw tool call: {tool_call}")
                print(f"   🎯 Function name: {tool_call['name']}")
                print(f"   📄 Arguments (JSON): {tool_call['arguments']}")
                print(f"   🆔 Call ID: {tool_call['id']}")
                
                # Validate before execution
                errors = executor.validate_tool_call(tool_call)
                if errors:
                    print(f"   ❌ Validation errors: {errors}")
                else:
                    print(f"   ✅ Validation: Passed")
                    
                    # Execute and show result
                    result = executor.execute(tool_call)
                    print(f"   🎯 Execution result: {result}")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "tool call details")


if __name__ == "__main__":
    demo_tool_execution()
    demo_tool_call_details() 