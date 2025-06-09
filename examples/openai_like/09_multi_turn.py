#!/usr/bin/env python3
"""
Example: Multi-Turn Conversations
Demonstrates: Extended conversations with context, tools, and reasoning

This example shows:
- Multi-turn conversation management
- Context preservation across turns
- Tools and reasoning in conversations
- Both streaming and non-streaming approaches
"""

from common import (
    get_default_provider, get_default_executor, 
    calculate, get_weather, search_web,
    print_section, print_subsection, handle_error, execute_tool_calls
)


def demo_basic_multi_turn():
    """Demonstrate basic multi-turn conversation."""
    
    print_section("Basic Multi-Turn Conversation")
    
    try:
        provider = get_default_provider()
        executor = get_default_executor()
        
        print("🤖 Starting multi-turn conversation...")
        print(f"   Model: {provider.model_id}")
        
        # Initialize conversation
        messages = []
        
        # Turn 1
        user_input = "Hi! I'm planning a trip to Japan. Can you help me with some questions?"
        messages.append({"role": "user", "content": user_input})
        
        print(f"\n👤 Turn 1: {user_input}")
        response = provider.chat(messages)
        print(f"🤖 Assistant: {response.content}")
        messages.append({"role": "assistant", "content": response.content})
        
        # Turn 2
        user_input = "What's the weather like in Tokyo today?"
        messages.append({"role": "user", "content": user_input})
        
        print(f"\n👤 Turn 2: {user_input}")
        response = provider.chat(messages)
        print(f"🤖 Assistant: {response.content}")
        messages.append({"role": "assistant", "content": response.content})
        
        # Turn 3
        user_input = "Thanks! Can you calculate how much 150 USD would be in Japanese Yen if the exchange rate is 1 USD = 149 JPY?"
        messages.append({"role": "user", "content": user_input})
        
        print(f"\n👤 Turn 3: {user_input}")
        response = provider.chat(messages)
        print(f"🤖 Assistant: {response.content}")
        messages.append({"role": "assistant", "content": response.content})
        
        # Turn 4 - Reference previous conversation
        user_input = "Going back to the weather you mentioned earlier, should I pack warm clothes?"
        messages.append({"role": "user", "content": user_input})
        
        print(f"\n👤 Turn 4: {user_input}")
        response = provider.chat(messages)
        print(f"🤖 Assistant: {response.content}")
        
        print(f"\n📊 Conversation Summary:")
        print(f"   Total turns: 4")
        print(f"   Total messages in context: {len(messages) + 1}")
        
        provider.close()
        
        print("\n✅ Basic multi-turn conversation completed!")
        
    except Exception as e:
        handle_error(e, "basic multi-turn")


def demo_multi_turn_with_tools():
    """Demonstrate multi-turn conversation with tools."""
    
    print_section("Multi-Turn with Tools")
    
    try:
        provider = get_default_provider(tools=[calculate, get_weather, search_web])
        executor = get_default_executor()
        
        print("🤖 Starting tool-enhanced conversation...")
        print(f"   Available tools: {[tool.__name__ for tool in provider.tools]}")
        
        messages = []
        
        def process_turn(user_input, turn_number):
            """Process a single conversation turn with potential tool calls."""
            messages.append({"role": "user", "content": user_input})
            
            print(f"\n👤 Turn {turn_number}: {user_input}")
            
            response = provider.chat(messages)
            print(f"🤖 Assistant: {response.content}")
            
            # Handle tool calls if present
            if response.tool_calls:
                print(f"🔧 Tool calls requested: {len(response.tool_calls)}")
                
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant", 
                    "content": response.content, 
                    "tool_calls": response.tool_calls
                })
                
                # Execute tools
                tool_results = execute_tool_calls(executor, response.tool_calls)
                messages.extend(tool_results)
                
                # Get final response
                final_response = provider.chat(messages)
                print(f"🤖 Assistant (after tools): {final_response.content}")
                messages.append({"role": "assistant", "content": final_response.content})
            else:
                messages.append({"role": "assistant", "content": response.content})
        
        # Conversation with tools
        process_turn("I need to plan a budget for my trip. What's 500 * 1.2?", 1)
        process_turn("What's the weather like in London?", 2)
        process_turn("Can you search for information about travel insurance?", 3)
        process_turn("Based on the weather in London you just checked, and the budget calculation of $600, what would you recommend?", 4)
        
        print(f"\n📊 Tool-Enhanced Conversation Summary:")
        print(f"   Total turns: 4")
        print(f"   Messages in context: {len(messages)}")
        print(f"   Context types: {set(msg['role'] for msg in messages)}")
        
        provider.close()
        
        print("\n✅ Multi-turn with tools completed!")
        
    except Exception as e:
        handle_error(e, "multi-turn with tools")


def demo_streaming_multi_turn():
    """Demonstrate multi-turn conversation with streaming."""
    
    print_section("Streaming Multi-Turn Conversation")
    
    try:
        provider = get_default_provider(tools=[calculate])
        executor = get_default_executor()
        
        print("🤖 Starting streaming multi-turn conversation...")
        
        messages = []
        
        def streaming_turn(user_input, turn_number):
            """Process a turn with streaming."""
            messages.append({"role": "user", "content": user_input})
            
            print(f"\n👤 Turn {turn_number}: {user_input}")
            print("🤖 Assistant: ", end="", flush=True)
            
            # Stream response
            full_content = ""
            accumulated_tool_calls = []
            
            for chunk in provider.stream(messages):
                if chunk.content_delta:
                    print(chunk.content_delta, end="", flush=True)
                    full_content += chunk.content_delta
                
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
                    break
            
            print()  # New line after streaming
            
            # Handle tool calls
            if accumulated_tool_calls:
                messages.append({
                    "role": "assistant", 
                    "content": full_content, 
                    "tool_calls": accumulated_tool_calls
                })
                
                print(f"🔧 Executing {len(accumulated_tool_calls)} tool calls...")
                tool_results = execute_tool_calls(executor, accumulated_tool_calls, verbose=False)
                messages.extend(tool_results)
                
                # Stream final response
                print("🤖 Assistant (continued): ", end="", flush=True)
                for chunk in provider.stream(messages):
                    if chunk.content_delta:
                        print(chunk.content_delta, end="", flush=True)
                    if chunk.finish_reason:
                        final_content = chunk.content if hasattr(chunk, 'content') else ""
                        break
                
                print()  # New line
                if 'final_content' in locals():
                    messages.append({"role": "assistant", "content": final_content})
            else:
                messages.append({"role": "assistant", "content": full_content})
        
        # Streaming conversation
        streaming_turn("Let's do some math. What's 25 * 30?", 1)
        streaming_turn("Now divide that result by 5", 2)
        streaming_turn("Perfect! Can you explain why that calculation sequence might be useful?", 3)
        
        print(f"\n📊 Streaming Conversation Summary:")
        print(f"   Total turns: 3")
        print(f"   Messages: {len(messages)}")
        
        provider.close()
        
        print("\n✅ Streaming multi-turn completed!")
        
    except Exception as e:
        handle_error(e, "streaming multi-turn")


def demo_conversation_with_reasoning():
    """Demonstrate multi-turn conversation with reasoning tracking."""
    
    print_section("Multi-Turn with Reasoning Tracking")
    
    try:
        provider = get_default_provider()
        
        print("🤖 Starting reasoning-tracked conversation...")
        
        messages = []
        reasoning_history = []
        
        def reasoning_turn(user_input, turn_number):
            """Process a turn while tracking reasoning."""
            messages.append({"role": "user", "content": user_input})
            
            print(f"\n👤 Turn {turn_number}: {user_input}")
            
            response = provider.chat(messages)
            print(f"🤖 Assistant: {response.content}")
            
            # Track reasoning
            if response.reasoning:
                print(f"🧠 Reasoning detected: {response.reasoning_format}")
                reasoning_history.append({
                    'turn': turn_number,
                    'format': response.reasoning_format,
                    'content': response.reasoning,
                    'length': len(response.reasoning)
                })
                
                # Show reasoning preview
                preview = response.reasoning[:100] + "..." if len(response.reasoning) > 100 else response.reasoning
                print(f"💭 Reasoning preview: {preview}")
            
            messages.append({"role": "assistant", "content": response.content})
        
        # Conversation that encourages reasoning
        reasoning_turn("I have a logic puzzle for you: If all birds can fly, and penguins are birds, but penguins can't fly, what's wrong with this reasoning?", 1)
        reasoning_turn("That's interesting! Now, if I told you that some birds are pets, and all pets are loved, can we conclude anything about some birds?", 2)
        reasoning_turn("Great reasoning! Can you now think through this: If it's raining, then the ground is wet. The ground is wet. Is it raining?", 3)
        
        # Show reasoning summary
        print(f"\n🧠 Reasoning Summary:")
        print(f"   Turns with reasoning: {len(reasoning_history)}")
        
        for entry in reasoning_history:
            print(f"   Turn {entry['turn']}: {entry['format']} ({entry['length']} chars)")
        
        total_reasoning_chars = sum(entry['length'] for entry in reasoning_history)
        print(f"   Total reasoning content: {total_reasoning_chars} characters")
        
        provider.close()
        
        print("\n✅ Reasoning-tracked conversation completed!")
        
    except Exception as e:
        handle_error(e, "conversation with reasoning")


def demo_context_management():
    """Demonstrate conversation context management strategies."""
    
    print_section("Context Management Strategies")
    
    try:
        provider = get_default_provider()
        
        print("🤖 Demonstrating context management...")
        
        # Strategy 1: Full context preservation
        print_subsection("Strategy 1: Full Context")
        
        full_messages = []
        for i in range(3):
            user_input = f"This is message {i+1}. Please remember this number."
            full_messages.append({"role": "user", "content": user_input})
            
            response = provider.chat(full_messages)
            full_messages.append({"role": "assistant", "content": response.content})
            
            print(f"Turn {i+1}: {len(full_messages)} messages in context")
        
        # Test context recall
        full_messages.append({"role": "user", "content": "What were all the numbers I asked you to remember?"})
        response = provider.chat(full_messages)
        print(f"Context recall: {response.content}")
        
        # Strategy 2: Context summarization (simulated)
        print_subsection("Strategy 2: Context Summarization")
        
        # Simulate summarization by keeping only recent messages
        def summarize_context(messages, keep_recent=4):
            """Simple context management: keep system message + recent messages."""
            if len(messages) <= keep_recent:
                return messages
            
            # Keep first message (if system) and last N messages
            summary = "Previous conversation covered: "
            for msg in messages[:-keep_recent]:
                if msg['role'] == 'user':
                    summary += f"User asked about {msg['content'][:30]}... "
            
            recent_messages = messages[-keep_recent:]
            return [{"role": "system", "content": summary}] + recent_messages
        
        summarized_messages = []
        for i in range(5):
            user_input = f"Question {i+1}: What's {i+1} * 10?"
            summarized_messages.append({"role": "user", "content": user_input})
            
            # Manage context
            context_to_use = summarize_context(summarized_messages)
            
            response = provider.chat(context_to_use)
            summarized_messages.append({"role": "assistant", "content": response.content})
            
            print(f"Turn {i+1}: {len(summarized_messages)} total, {len(context_to_use)} in context")
        
        print(f"\n📊 Context Management Comparison:")
        print(f"   Full context final size: {len(full_messages)} messages")
        print(f"   Summarized context final size: {len(summarized_messages)} messages")
        print(f"   Context efficiency: {len(summarized_messages)/len(full_messages):.1%}")
        
        provider.close()
        
        print("\n✅ Context management demo completed!")
        
    except Exception as e:
        handle_error(e, "context management")


if __name__ == "__main__":
    demo_basic_multi_turn()
    demo_multi_turn_with_tools()
    demo_streaming_multi_turn()
    demo_conversation_with_reasoning()
    demo_context_management() 