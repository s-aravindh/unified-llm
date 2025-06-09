#!/usr/bin/env python3
"""
Example: Streaming Reasoning
Demonstrates: Reasoning extraction during streaming responses

This example shows:
- Real-time reasoning delta processing
- Streaming reasoning vs content separation
- Reasoning pattern detection in streams
- Different reasoning formats during streaming
"""

import time
from common import get_default_provider, print_section, handle_error


def demo_streaming_reasoning():
    """Demonstrate reasoning extraction during streaming."""
    
    print_section("Streaming Reasoning Example")
    
    try:
        provider = get_default_provider()
        
        print("🤖 Initializing streaming provider for reasoning...")
        print(f"   Model: {provider.model_id}")
        
        messages = [
            {
                "role": "user", 
                "content": "Think step by step: If I have 3 boxes, and each box contains 4 smaller boxes, and each smaller box contains 6 items, how many items do I have in total? Show your reasoning process."
            }
        ]
        
        print(f"\n👤 User: {messages[0]['content']}")
        print("🤖 Assistant reasoning...")
        
        # Stream and separate reasoning from content
        content_parts = []
        reasoning_parts = []
        
        for chunk in provider.stream(messages):
            # Handle reasoning delta
            if chunk.reasoning_delta:
                print(f"🧠 Reasoning: {chunk.reasoning_delta}", end="", flush=True)
                reasoning_parts.append(chunk.reasoning_delta)
            
            # Handle content delta
            if chunk.content_delta:
                print(f"💬 Content: {chunk.content_delta}", end="", flush=True)
                content_parts.append(chunk.content_delta)
            
            if chunk.finish_reason:
                break
        
        # Show final results
        full_content = "".join(content_parts)
        full_reasoning = "".join(reasoning_parts)
        
        print(f"\n\n📊 Streaming Results:")
        print(f"   Content chunks: {len(content_parts)}")
        print(f"   Reasoning chunks: {len(reasoning_parts)}")
        print(f"   Total content: {len(full_content)} characters")
        print(f"   Total reasoning: {len(full_reasoning)} characters")
        
        if full_reasoning:
            print(f"\n🧠 Complete Reasoning:")
            print("-" * 50)
            print(full_reasoning)
            print("-" * 50)
        
        if full_content:
            print(f"\n💬 Complete Content:")
            print("-" * 50)
            print(full_content)
            print("-" * 50)
        
        provider.close()
        
        print("\n✅ Streaming reasoning example completed!")
        
    except Exception as e:
        handle_error(e, "streaming reasoning")


def demo_reasoning_pattern_detection():
    """Demonstrate pattern-based reasoning detection during streaming."""
    
    print_section("Reasoning Pattern Detection During Streaming")
    
    try:
        provider = get_default_provider()
        
        messages = [
            {
                "role": "user",
                "content": "Solve this logic puzzle: All cats are animals. Some animals are pets. Can we conclude that some cats are pets? Think through this carefully."
            }
        ]
        
        print(f"👤 User: {messages[0]['content']}")
        print("🤖 Analyzing stream for reasoning patterns...")
        
        # Track pattern detection state
        in_reasoning_block = False
        reasoning_buffer = ""
        content_buffer = ""
        reasoning_patterns = ["<think>", "<thinking>", "<reason>", "<analysis>"]
        closing_patterns = ["</think>", "</thinking>", "</reason>", "</analysis>"]
        
        for chunk in provider.stream(messages):
            if chunk.content_delta:
                text = chunk.content_delta
                
                # Check for reasoning pattern start
                if not in_reasoning_block:
                    for pattern in reasoning_patterns:
                        if pattern in text.lower():
                            in_reasoning_block = True
                            print(f"\n🧠 Reasoning block detected: {pattern}")
                            # Split at pattern
                            parts = text.lower().split(pattern, 1)
                            if len(parts) > 1:
                                content_buffer += parts[0]
                                reasoning_buffer += parts[1]
                            break
                    else:
                        # No pattern found, add to content
                        content_buffer += text
                        print(f"💬 Content: {text}", end="", flush=True)
                else:
                    # Currently in reasoning block
                    reasoning_buffer += text
                    print(f"🧠 Reasoning: {text}", end="", flush=True)
                    
                    # Check for closing pattern
                    for pattern in closing_patterns:
                        if pattern in text.lower():
                            in_reasoning_block = False
                            print(f"\n✅ Reasoning block ended: {pattern}")
                            # Split at closing pattern
                            parts = text.lower().split(pattern, 1)
                            if len(parts) > 1:
                                content_buffer += parts[1]
                            break
            
            if chunk.finish_reason:
                break
        
        print(f"\n\n📊 Pattern Detection Results:")
        print(f"   Final state: {'In reasoning' if in_reasoning_block else 'In content'}")
        print(f"   Content buffer: {len(content_buffer)} characters")
        print(f"   Reasoning buffer: {len(reasoning_buffer)} characters")
        
        if reasoning_buffer:
            print(f"\n🧠 Extracted Reasoning:")
            print(reasoning_buffer.strip())
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "reasoning pattern detection")


def demo_mixed_streaming():
    """Demonstrate mixed content and reasoning streaming."""
    
    print_section("Mixed Content and Reasoning Streaming")
    
    try:
        provider = get_default_provider()
        
        messages = [
            {
                "role": "user",
                "content": "Calculate the compound interest on $1000 at 5% annually for 3 years. Think through the formula and show your work step by step."
            }
        ]
        
        print(f"👤 User: {messages[0]['content']}")
        print("🤖 Assistant with mixed streaming...")
        
        # Track different types of content
        content_timeline = []
        
        for chunk in provider.stream(messages):
            timestamp = time.time()
            
            if chunk.content_delta:
                content_timeline.append({
                    "time": timestamp,
                    "type": "content",
                    "text": chunk.content_delta
                })
                print(f"💬 {chunk.content_delta}", end="", flush=True)
            
            if chunk.reasoning_delta:
                content_timeline.append({
                    "time": timestamp,
                    "type": "reasoning", 
                    "text": chunk.reasoning_delta
                })
                print(f"\n🧠 [Reasoning] {chunk.reasoning_delta}", end="", flush=True)
            
            if chunk.finish_reason:
                break
        
        print(f"\n")
        
        # Analyze timeline
        if content_timeline:
            print(f"\n📈 Content Timeline Analysis:")
            print(f"   Total chunks: {len(content_timeline)}")
            
            content_chunks = [item for item in content_timeline if item["type"] == "content"]
            reasoning_chunks = [item for item in content_timeline if item["type"] == "reasoning"]
            
            print(f"   Content chunks: {len(content_chunks)}")
            print(f"   Reasoning chunks: {len(reasoning_chunks)}")
            
            if content_timeline:
                start_time = content_timeline[0]["time"]
                for i, item in enumerate(content_timeline[:5]):  # Show first 5 items
                    elapsed = item["time"] - start_time
                    print(f"   {i+1}. [{elapsed:.2f}s] {item['type']}: {item['text'][:30]}...")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "mixed streaming")


def demo_reasoning_vs_content_timing():
    """Demonstrate timing differences between reasoning and content streaming."""
    
    print_section("Reasoning vs Content Timing Analysis")
    
    try:
        provider = get_default_provider()
        
        messages = [
            {
                "role": "user",
                "content": "Explain the Pythagorean theorem and then calculate the hypotenuse of a right triangle with sides 3 and 4. Think through both the explanation and the calculation."
            }
        ]
        
        print(f"👤 User: {messages[0]['content']}")
        
        # Timing tracking
        start_time = time.time()
        first_content_time = None
        first_reasoning_time = None
        content_count = 0
        reasoning_count = 0
        
        print("🤖 Streaming with timing analysis...")
        
        for chunk in provider.stream(messages):
            current_time = time.time() - start_time
            
            if chunk.content_delta:
                if first_content_time is None:
                    first_content_time = current_time
                    print(f"\n⏱️  First content at {current_time:.2f}s")
                
                content_count += 1
                print(f"💬 [{current_time:.2f}s] {chunk.content_delta}", end="", flush=True)
            
            if chunk.reasoning_delta:
                if first_reasoning_time is None:
                    first_reasoning_time = current_time
                    print(f"\n⏱️  First reasoning at {current_time:.2f}s")
                
                reasoning_count += 1
                print(f"\n🧠 [{current_time:.2f}s] {chunk.reasoning_delta}", end="", flush=True)
            
            if chunk.finish_reason:
                final_time = current_time
                break
        
        print(f"\n")
        
        # Show timing analysis
        print(f"\n📊 Timing Analysis:")
        print(f"   Total streaming time: {final_time:.2f}s")
        if first_content_time:
            print(f"   Time to first content: {first_content_time:.2f}s")
        if first_reasoning_time:
            print(f"   Time to first reasoning: {first_reasoning_time:.2f}s")
        print(f"   Content chunks: {content_count}")
        print(f"   Reasoning chunks: {reasoning_count}")
        
        if first_content_time and first_reasoning_time:
            if first_reasoning_time < first_content_time:
                print(f"   📝 Reasoning came first (by {first_content_time - first_reasoning_time:.2f}s)")
            else:
                print(f"   📝 Content came first (by {first_reasoning_time - first_content_time:.2f}s)")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "reasoning vs content timing")


if __name__ == "__main__":
    demo_streaming_reasoning()
    demo_reasoning_pattern_detection()
    demo_mixed_streaming()
    demo_reasoning_vs_content_timing() 