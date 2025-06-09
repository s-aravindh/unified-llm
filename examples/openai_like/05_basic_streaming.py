#!/usr/bin/env python3
"""
Example: Basic Streaming
Demonstrates: Real-time streaming responses without tools

This example shows:
- Basic streaming setup
- Token-by-token response handling
- Stream completion detection
- Error handling in streaming mode
"""

import time
from common import get_default_provider, print_section, handle_error


def demo_basic_streaming():
    """Demonstrate basic streaming functionality."""
    
    print_section("Basic Streaming Example")
    
    try:
        provider = get_default_provider()
        
        print("🤖 Initializing streaming provider...")
        print(f"   Model: {provider.model_id}")
        
        messages = [
            {"role": "user", "content": "Tell me a short story about a robot learning to paint. Make it engaging and creative."}
        ]
        
        print(f"\n👤 User: {messages[0]['content']}")
        print("🤖 Assistant: ", end="", flush=True)
        
        # Stream the response
        full_content = ""
        start_time = time.time()
        token_count = 0
        
        for chunk in provider.stream(messages):
            if chunk.content_delta:
                print(chunk.content_delta, end="", flush=True)
                full_content += chunk.content_delta
                token_count += 1
            
            # Handle stream completion
            if chunk.finish_reason:
                break
        
        end_time = time.time()
        
        # Show streaming statistics
        print(f"\n\n📊 Streaming Statistics:")
        print(f"   Total time: {end_time - start_time:.2f}s")
        print(f"   Approximate tokens: {token_count}")
        print(f"   Characters: {len(full_content)}")
        print(f"   Finish reason: {chunk.finish_reason if 'chunk' in locals() else 'unknown'}")
        
        if end_time - start_time > 0:
            tokens_per_sec = token_count / (end_time - start_time)
            print(f"   Streaming speed: {tokens_per_sec:.1f} tokens/sec")
        
        provider.close()
        
        print("\n✅ Basic streaming example completed!")
        
    except Exception as e:
        handle_error(e, "basic streaming")


def demo_streaming_vs_non_streaming():
    """Compare streaming vs non-streaming performance."""
    
    print_section("Streaming vs Non-Streaming Comparison")
    
    try:
        provider = get_default_provider()
        
        query = "Explain the concept of machine learning in simple terms, covering supervised learning, unsupervised learning, and reinforcement learning."
        messages = [{"role": "user", "content": query}]
        
        print(f"📝 Test query: {query}")
        
        # Non-streaming approach
        print(f"\n🔄 Method 1: Non-Streaming")
        start_time = time.time()
        
        response = provider.chat(messages)
        
        non_streaming_time = time.time() - start_time
        print(f"   Response received in: {non_streaming_time:.2f}s")
        print(f"   Content length: {len(response.content)} characters")
        print(f"   First 100 chars: {response.content[:100]}...")
        
        # Streaming approach
        print(f"\n⚡ Method 2: Streaming")
        start_time = time.time()
        first_token_time = None
        
        print("   Response: ", end="", flush=True)
        full_content = ""
        chunk_count = 0
        
        for chunk in provider.stream(messages):
            if chunk.content_delta:
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                
                print(chunk.content_delta, end="", flush=True)
                full_content += chunk.content_delta
                chunk_count += 1
            
            if chunk.finish_reason:
                break
        
        streaming_time = time.time() - start_time
        
        print(f"\n\n📊 Performance Comparison:")
        print(f"   Non-streaming total time: {non_streaming_time:.2f}s")
        print(f"   Streaming total time: {streaming_time:.2f}s")
        if first_token_time:
            print(f"   Time to first token: {first_token_time:.2f}s")
        print(f"   Streaming chunks: {chunk_count}")
        print(f"   Content length match: {'✅' if len(response.content) == len(full_content) else '❌'}")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "streaming comparison")


def demo_streaming_interruption():
    """Demonstrate streaming interruption and control."""
    
    print_section("Streaming Interruption Control")
    
    try:
        provider = get_default_provider()
        
        messages = [
            {"role": "user", "content": "Write a detailed explanation of quantum computing, covering quantum bits, superposition, entanglement, and quantum algorithms. Make it comprehensive."}
        ]
        
        print(f"👤 User: {messages[0]['content']}")
        print("🤖 Assistant: ", end="", flush=True)
        
        full_content = ""
        char_count = 0
        max_chars = 200  # Interrupt after 200 characters
        
        for chunk in provider.stream(messages):
            if chunk.content_delta:
                print(chunk.content_delta, end="", flush=True)
                full_content += chunk.content_delta
                char_count += len(chunk.content_delta)
                
                # Simulate interruption condition
                if char_count >= max_chars:
                    print(f"\n\n⏹️  Interrupting stream at {char_count} characters...")
                    break
            
            if chunk.finish_reason:
                print(f"\n\n✅ Stream completed naturally")
                break
        
        print(f"\n📊 Interruption Results:")
        print(f"   Characters received: {char_count}")
        print(f"   Interruption triggered: {'Yes' if char_count >= max_chars else 'No'}")
        print(f"   Partial content: {full_content}")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "streaming interruption")


def demo_streaming_with_delays():
    """Demonstrate streaming with artificial delays to show real-time nature."""
    
    print_section("Streaming with Artificial Delays")
    
    try:
        provider = get_default_provider()
        
        messages = [
            {"role": "user", "content": "Count from 1 to 10 and briefly describe each number."}
        ]
        
        print(f"👤 User: {messages[0]['content']}")
        print("🤖 Assistant: ", end="", flush=True)
        
        word_count = 0
        
        for chunk in provider.stream(messages):
            if chunk.content_delta:
                print(chunk.content_delta, end="", flush=True)
                
                # Add small delay every few words to demonstrate real-time nature
                if ' ' in chunk.content_delta:
                    word_count += chunk.content_delta.count(' ')
                    if word_count % 5 == 0:  # Every 5 words
                        time.sleep(0.1)  # Small delay
            
            if chunk.finish_reason:
                break
        
        print(f"\n\n✅ Streaming with delays completed!")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "streaming with delays")


if __name__ == "__main__":
    demo_basic_streaming()
    demo_streaming_vs_non_streaming()
    demo_streaming_interruption()
    demo_streaming_with_delays() 