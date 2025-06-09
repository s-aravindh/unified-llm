#!/usr/bin/env python3
"""
Example: Reasoning
Demonstrates: Reasoning extraction from various formats

This example shows:
- OpenAI o1 style reasoning tokens
- vLLM native reasoning fields
- Pattern-based reasoning (<think>, <reasoning> tags)
- Reasoning token counting and metadata
- Different reasoning formats across providers
"""

from common import get_default_provider, print_section, handle_error


def demo_reasoning_extraction():
    """Demonstrate reasoning extraction from responses."""
    
    print_section("Reasoning Extraction Example")
    
    try:
        # Initialize provider - reasoning works with most models
        provider = get_default_provider()
        
        print("🤖 Initializing provider for reasoning tasks...")
        print(f"   Model: {provider.model_id}")
        
        # Query that encourages reasoning
        messages = [
            {
                "role": "user", 
                "content": "Solve this step by step: If a train travels 120km in 2 hours, and then 180km in the next 3 hours, what is its average speed for the entire journey? Please show your thinking process."
            }
        ]
        
        print(f"\n👤 User: {messages[0]['content']}")
        print("🤖 Assistant reasoning...")
        
        # Get response with potential reasoning
        response = provider.chat(messages)
        
        # Display main content
        print(f"🤖 Assistant: {response.content}")
        
        # Check for reasoning
        if response.reasoning:
            print(f"\n🧠 Reasoning detected!")
            print(f"   Format: {response.reasoning_format}")
            print(f"   Length: {len(response.reasoning)} characters")
            
            # Show reasoning content
            print(f"\n💭 Reasoning content:")
            print("-" * 50)
            print(response.reasoning)
            print("-" * 50)
            
            # Show metadata if available
            if response.metadata and 'reasoning_tokens' in response.metadata:
                print(f"\n📊 Reasoning metadata:")
                print(f"   Reasoning tokens: {response.metadata['reasoning_tokens']}")
                if 'total_tokens' in response.metadata:
                    total = response.metadata['total_tokens']
                    reasoning = response.metadata['reasoning_tokens']
                    print(f"   Content tokens: {total - reasoning}")
                    print(f"   Reasoning ratio: {reasoning/total:.1%}")
        else:
            print(f"\nℹ️ No explicit reasoning detected")
            print(f"   The model may have reasoned implicitly in the response")
        
        provider.close()
        
        print("\n✅ Reasoning extraction example completed!")
        
    except Exception as e:
        handle_error(e, "reasoning extraction")


def demo_reasoning_formats():
    """Demonstrate different reasoning formats."""
    
    print_section("Different Reasoning Formats")
    
    try:
        provider = get_default_provider()
        
        # Test different prompting styles that might trigger different reasoning formats
        test_cases = [
            {
                "name": "Explicit reasoning request",
                "message": "Think step by step: What's 15% of 240?"
            },
            {
                "name": "Problem-solving request", 
                "message": "I need to solve this problem: A recipe calls for 3 cups of flour for 12 cookies. How much flour for 20 cookies? Show your work."
            },
            {
                "name": "Analysis request",
                "message": "Analyze this: If I save $50 per month, how long will it take to save $600? Break down your thinking."
            }
        ]
        
        for test_case in test_cases:
            print(f"\n📝 Test: {test_case['name']}")
            print(f"👤 Query: {test_case['message']}")
            
            messages = [{"role": "user", "content": test_case['message']}]
            response = provider.chat(messages)
            
            print(f"🤖 Response: {response.content[:100]}{'...' if len(response.content) > 100 else ''}")
            
            if response.reasoning:
                print(f"🧠 Reasoning found: {response.reasoning_format}")
                print(f"   Length: {len(response.reasoning)} chars")
                
                # Show first few lines of reasoning
                reasoning_lines = response.reasoning.split('\n')[:3]
                print(f"   Preview: {reasoning_lines[0][:60]}{'...' if len(reasoning_lines[0]) > 60 else ''}")
            else:
                print(f"ℹ️ No reasoning detected")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "reasoning formats")


def demo_reasoning_with_tools():
    """Demonstrate reasoning combined with tool usage."""
    
    print_section("Reasoning with Tools")
    
    try:
        from common import calculate, get_default_executor
        
        provider = get_default_provider(tools=[calculate])
        executor = get_default_executor()
        
        messages = [
            {
                "role": "user",
                "content": "I need to calculate the area of a room that's 12.5 feet by 8.3 feet, then convert that to square meters (1 square foot = 0.092903 square meters). Think through this step by step."
            }
        ]
        
        print(f"👤 User: {messages[0]['content']}")
        
        response = provider.chat(messages)
        
        print(f"🤖 Assistant: {response.content}")
        
        # Show reasoning if present
        if response.reasoning:
            print(f"\n🧠 Assistant's reasoning:")
            print(response.reasoning)
        
        # Handle tool calls
        if response.tool_calls:
            print(f"\n🔧 Tool calls: {len(response.tool_calls)}")
            
            tool_results = []
            for tool_call in response.tool_calls:
                print(f"   Calling: {tool_call['name']}({tool_call['arguments']})")
                result = executor.execute(tool_call)
                print(f"   Result: {result}")
                
                tool_results.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call["id"]
                })
            
            # Continue conversation
            messages.extend([
                {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls},
                *tool_results
            ])
            
            final_response = provider.chat(messages)
            print(f"\n🤖 Final response: {final_response.content}")
            
            if final_response.reasoning:
                print(f"\n🧠 Final reasoning:")
                print(final_response.reasoning)
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "reasoning with tools")


def demo_reasoning_metadata():
    """Demonstrate reasoning metadata analysis."""
    
    print_section("Reasoning Metadata Analysis")
    
    try:
        provider = get_default_provider()
        
        # Complex problem that should trigger reasoning
        messages = [
            {
                "role": "user",
                "content": "Explain the logic behind this: If all roses are flowers, and some flowers are red, can we conclude that some roses are red? Walk through the logical reasoning."
            }
        ]
        
        response = provider.chat(messages)
        
        print(f"🤖 Response: {response.content}")
        
        # Analyze all available data
        print(f"\n📊 Complete Response Analysis:")
        print(f"   Content length: {len(response.content)} characters")
        print(f"   Reasoning present: {'Yes' if response.reasoning else 'No'}")
        
        if response.reasoning:
            print(f"   Reasoning format: {response.reasoning_format}")
            print(f"   Reasoning length: {len(response.reasoning)} characters")
            
            # Calculate reasoning ratio
            total_chars = len(response.content) + len(response.reasoning)
            reasoning_ratio = len(response.reasoning) / total_chars
            print(f"   Reasoning ratio: {reasoning_ratio:.1%}")
        
        # Show metadata
        if response.metadata:
            print(f"\n📋 Response Metadata:")
            for key, value in response.metadata.items():
                print(f"   {key}: {value}")
        else:
            print(f"\nℹ️ No metadata available")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "reasoning metadata")


if __name__ == "__main__":
    demo_reasoning_extraction()
    demo_reasoning_formats()
    demo_reasoning_with_tools() 
    demo_reasoning_metadata() 