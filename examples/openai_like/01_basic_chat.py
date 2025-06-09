#!/usr/bin/env python3
"""
Example: Basic Chat
Demonstrates: Simple conversation without tools or streaming

This example shows the most basic usage of the unified LLM interface:
- Initialize a provider
- Send a simple message
- Receive a response
- Handle basic errors
"""

from common import get_default_provider, print_section, handle_error


def demo_basic_chat():
    """Demonstrate basic chat functionality without tools."""
    
    print_section("Basic Chat Example")
    
    try:
        # Initialize provider without tools
        provider = get_default_provider()
        
        print("🤖 Initializing provider...")
        print(f"   Model: {provider.model_id}")
        print(f"   Endpoint: {provider.base_url}")
        
        # Simple conversation
        messages = [
            {"role": "user", "content": "Hello! Can you introduce yourself briefly?"}
        ]
        
        print(f"\n👤 User: {messages[0]['content']}")
        print("🤖 Assistant: ", end="", flush=True)
        
        # Get response
        response = provider.chat(messages)
        print(response.content)
        
        # Show metadata if available
        if response.metadata:
            print(f"\n📊 Response metadata:")
            if 'model' in response.metadata:
                print(f"   Model used: {response.metadata['model']}")
            if 'usage' in response.metadata:
                usage = response.metadata['usage']
                if usage:
                    print(f"   Tokens used: {usage}")
        
        # Second message to show conversation continuation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": "What can you help me with?"})
        
        print(f"\n👤 User: {messages[-1]['content']}")
        print("🤖 Assistant: ", end="", flush=True)
        
        response2 = provider.chat(messages)
        print(response2.content)
        
        provider.close()
        
        print("\n✅ Basic chat example completed successfully!")
        
    except Exception as e:
        handle_error(e, "basic chat")


if __name__ == "__main__":
    demo_basic_chat() 