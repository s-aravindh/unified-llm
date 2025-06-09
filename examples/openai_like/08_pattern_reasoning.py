#!/usr/bin/env python3
"""
Example: Pattern Reasoning
Demonstrates: Tag-based reasoning pattern extraction

This example shows:
- Detection of <think>, <reasoning>, <analysis> tags
- Pattern-based reasoning extraction from responses
- Different reasoning tag formats
- Reasoning content validation and processing
"""

import re
from common import get_default_provider, print_section, handle_error


def demo_pattern_reasoning():
    """Demonstrate pattern-based reasoning extraction."""
    
    print_section("Pattern-Based Reasoning Example")
    
    try:
        provider = get_default_provider()
        
        print("🤖 Initializing provider for pattern reasoning...")
        print(f"   Model: {provider.model_id}")
        
        # Query that might encourage reasoning tags
        messages = [
            {
                "role": "user", 
                "content": "Think through this problem step by step: A snail is at the bottom of a 20-foot well. Each day it climbs up 3 feet, but each night it slides back 2 feet. How many days will it take to reach the top? Use <think> tags to show your reasoning."
            }
        ]
        
        print(f"\n👤 User: {messages[0]['content']}")
        print("🤖 Assistant thinking...")
        
        response = provider.chat(messages)
        print(f"🤖 Assistant: {response.content}")
        
        # Extract reasoning using pattern matching
        reasoning_patterns = [
            (r'<think>(.*?)</think>', 'think'),
            (r'<thinking>(.*?)</thinking>', 'thinking'),
            (r'<reason>(.*?)</reason>', 'reason'),
            (r'<reasoning>(.*?)</reasoning>', 'reasoning'),
            (r'<analysis>(.*?)</analysis>', 'analysis'),
        ]
        
        found_reasoning = []
        
        for pattern, tag_name in reasoning_patterns:
            matches = re.findall(pattern, response.content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                found_reasoning.append({
                    'tag': tag_name,
                    'content': match.strip(),
                    'length': len(match.strip())
                })
        
        # Show extracted reasoning
        if found_reasoning:
            print(f"\n🧠 Pattern-based reasoning found: {len(found_reasoning)} blocks")
            
            for i, reasoning in enumerate(found_reasoning, 1):
                print(f"\n📋 Reasoning Block {i}:")
                print(f"   Tag: <{reasoning['tag']}>")
                print(f"   Length: {reasoning['length']} characters")
                print(f"   Content:")
                print("-" * 40)
                print(reasoning['content'])
                print("-" * 40)
        else:
            print(f"\nℹ️ No reasoning patterns found in response")
            
            # Check if reasoning was provided through provider's reasoning extraction
            if response.reasoning:
                print(f"✅ But reasoning was extracted by provider: {response.reasoning_format}")
                print(f"   Length: {len(response.reasoning)} characters")
        
        provider.close()
        
        print("\n✅ Pattern reasoning example completed!")
        
    except Exception as e:
        handle_error(e, "pattern reasoning")


def demo_multiple_reasoning_patterns():
    """Demonstrate multiple reasoning patterns in one response."""
    
    print_section("Multiple Reasoning Patterns")
    
    try:
        provider = get_default_provider()
        
        messages = [
            {
                "role": "user",
                "content": "Solve this complex problem: A store offers a 20% discount, then adds 8% tax. Is this better or worse than adding 8% tax first, then applying a 20% discount? Use different reasoning sections: <analysis> for initial thoughts, <thinking> for calculations, and <conclusion> for the final answer."
            }
        ]
        
        print(f"👤 User: {messages[0]['content']}")
        
        response = provider.chat(messages)
        print(f"🤖 Assistant: {response.content}")
        
        # More comprehensive pattern extraction
        patterns = {
            'analysis': r'<analysis>(.*?)</analysis>',
            'thinking': r'<thinking>(.*?)</thinking>',
            'think': r'<think>(.*?)</think>',
            'reasoning': r'<reasoning>(.*?)</reasoning>',
            'conclusion': r'<conclusion>(.*?)</conclusion>',
            'solution': r'<solution>(.*?)</solution>',
            'explanation': r'<explanation>(.*?)</explanation>',
        }
        
        extracted_sections = {}
        
        for section_name, pattern in patterns.items():
            matches = re.findall(pattern, response.content, re.DOTALL | re.IGNORECASE)
            if matches:
                extracted_sections[section_name] = matches
        
        # Display results
        if extracted_sections:
            print(f"\n📊 Reasoning Sections Found: {len(extracted_sections)}")
            
            for section_name, content_list in extracted_sections.items():
                print(f"\n🏷️  {section_name.title()} Section(s): {len(content_list)}")
                
                for i, content in enumerate(content_list, 1):
                    print(f"   {i}. Length: {len(content.strip())} characters")
                    
                    # Show preview
                    preview = content.strip()[:100]
                    if len(content.strip()) > 100:
                        preview += "..."
                    print(f"      Preview: {preview}")
        else:
            print(f"\nℹ️ No specific reasoning patterns found")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "multiple reasoning patterns")


def demo_reasoning_pattern_validation():
    """Demonstrate validation of reasoning patterns."""
    
    print_section("Reasoning Pattern Validation")
    
    try:
        provider = get_default_provider()
        
        messages = [
            {
                "role": "user",
                "content": "Explain the birthday paradox: In a room of 23 people, what's the probability that two people share the same birthday? Think through this mathematically and show your work."
            }
        ]
        
        response = provider.chat(messages)
        print(f"🤖 Assistant: {response.content}")
        
        # Comprehensive pattern analysis
        def analyze_reasoning_patterns(text):
            results = {
                'patterns_found': [],
                'malformed_tags': [],
                'nested_tags': [],
                'valid_reasoning': []
            }
            
            # Common reasoning tags
            reasoning_tags = ['think', 'thinking', 'reason', 'reasoning', 'analysis', 'solution']
            
            for tag in reasoning_tags:
                # Look for opening tags
                opening_pattern = f'<{tag}>'
                closing_pattern = f'</{tag}>'
                
                # Find all occurrences
                open_positions = [m.start() for m in re.finditer(re.escape(opening_pattern), text, re.IGNORECASE)]
                close_positions = [m.start() for m in re.finditer(re.escape(closing_pattern), text, re.IGNORECASE)]
                
                if open_positions:
                    results['patterns_found'].append(tag)
                    
                    # Check for proper pairing
                    if len(open_positions) == len(close_positions):
                        for i, open_pos in enumerate(open_positions):
                            if i < len(close_positions):
                                close_pos = close_positions[i]
                                if close_pos > open_pos:
                                    content = text[open_pos + len(opening_pattern):close_pos]
                                    results['valid_reasoning'].append({
                                        'tag': tag,
                                        'content': content.strip(),
                                        'start': open_pos,
                                        'end': close_pos + len(closing_pattern)
                                    })
                                else:
                                    results['malformed_tags'].append(f"Closing tag before opening for {tag}")
                    else:
                        results['malformed_tags'].append(f"Mismatched {tag} tags: {len(open_positions)} open, {len(close_positions)} close")
            
            return results
        
        analysis = analyze_reasoning_patterns(response.content)
        
        print(f"\n📊 Pattern Analysis Results:")
        print(f"   Patterns found: {analysis['patterns_found']}")
        print(f"   Valid reasoning blocks: {len(analysis['valid_reasoning'])}")
        print(f"   Malformed tags: {len(analysis['malformed_tags'])}")
        
        if analysis['valid_reasoning']:
            print(f"\n✅ Valid Reasoning Blocks:")
            for i, block in enumerate(analysis['valid_reasoning'], 1):
                print(f"   {i}. Tag: <{block['tag']}>")
                print(f"      Position: {block['start']}-{block['end']}")
                print(f"      Length: {len(block['content'])} characters")
                print(f"      Preview: {block['content'][:50]}{'...' if len(block['content']) > 50 else ''}")
        
        if analysis['malformed_tags']:
            print(f"\n❌ Malformed Tags:")
            for issue in analysis['malformed_tags']:
                print(f"   - {issue}")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "reasoning pattern validation")


def demo_custom_reasoning_extraction():
    """Demonstrate custom reasoning extraction with flexible patterns."""
    
    print_section("Custom Reasoning Extraction")
    
    try:
        provider = get_default_provider()
        
        messages = [
            {
                "role": "user",
                "content": "Design a solution for this: How would you organize a library's book classification system? Consider efficiency, user experience, and scalability. Feel free to structure your thoughts however you prefer."
            }
        ]
        
        response = provider.chat(messages)
        print(f"🤖 Assistant: {response.content}")
        
        # Custom extraction function
        def extract_structured_thinking(text):
            # Patterns for different thinking structures
            patterns = {
                'explicit_tags': {
                    'pattern': r'<(\w+)>(.*?)</\1>',
                    'description': 'Explicit XML-style reasoning tags'
                },
                'step_patterns': {
                    'pattern': r'(?:Step \d+|Phase \d+|Stage \d+):\s*(.*?)(?=(?:Step \d+|Phase \d+|Stage \d+)|$)',
                    'description': 'Step-by-step reasoning'
                },
                'bullet_thoughts': {
                    'pattern': r'(?:^|\n)[-*•]\s+(.*?)(?=\n|$)',
                    'description': 'Bullet-point thoughts'
                },
                'numbered_points': {
                    'pattern': r'(?:^|\n)\d+\.\s+(.*?)(?=\n|$)',
                    'description': 'Numbered reasoning points'
                },
                'consideration_blocks': {
                    'pattern': r'(?:Consider|Considering|Thought|Thinking about):\s*(.*?)(?=\n\n|$)',
                    'description': 'Consideration blocks'
                }
            }
            
            results = {}
            
            for pattern_name, pattern_info in patterns.items():
                matches = re.findall(pattern_info['pattern'], text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
                if matches:
                    results[pattern_name] = {
                        'matches': matches,
                        'count': len(matches),
                        'description': pattern_info['description']
                    }
            
            return results
        
        structured_thinking = extract_structured_thinking(response.content)
        
        print(f"\n🎯 Structured Thinking Analysis:")
        
        if structured_thinking:
            for pattern_name, data in structured_thinking.items():
                print(f"\n📋 {data['description']}:")
                print(f"   Found: {data['count']} instances")
                
                # Show first few examples
                for i, match in enumerate(data['matches'][:3], 1):
                    if isinstance(match, tuple):  # For patterns that capture groups
                        content = ' '.join(match) if match else ''
                    else:
                        content = match
                    
                    preview = content.strip()[:60]
                    if len(content.strip()) > 60:
                        preview += "..."
                    print(f"   {i}. {preview}")
                
                if data['count'] > 3:
                    print(f"   ... and {data['count'] - 3} more")
        else:
            print("   No structured thinking patterns detected")
            print("   Content appears to be in free-form narrative style")
        
        provider.close()
        
    except Exception as e:
        handle_error(e, "custom reasoning extraction")


if __name__ == "__main__":
    demo_pattern_reasoning()
    demo_multiple_reasoning_patterns()
    demo_reasoning_pattern_validation()
    demo_custom_reasoning_extraction() 