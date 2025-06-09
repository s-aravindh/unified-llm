#!/usr/bin/env python3
"""
Main Example Runner for OpenAI-like Provider
Interactive menu to run different examples and demonstrations.
"""

import sys
import importlib
from common import print_section, handle_error


EXAMPLES = {
    "Core Functionality": {
        "01_basic_chat": "Basic Chat - Simple conversation without tools",
        "02_tool_execution": "Tool Execution - Single tool call execution",
        "03_multiple_tools": "Multiple Tools - Multiple tool calls in one response", 
        "04_reasoning": "Reasoning - Reasoning extraction from responses"
    },
    "Streaming": {
        "05_basic_streaming": "Basic Streaming - Real-time response streaming",
        "06_streaming_with_tools": "Streaming with Tools - Tool calls during streaming",
        "07_streaming_reasoning": "Streaming Reasoning - Reasoning extraction while streaming",
        "08_pattern_reasoning": "Pattern Reasoning - Tag-based reasoning patterns"
    },
    "Conversation Management": {
        "09_multi_turn": "Multi-Turn - Extended conversations with context",
        "10_complex_workflow": "Complex Workflow - Advanced interaction patterns"
    }
}


def show_menu():
    """Display the interactive menu."""
    print_section("Unified LLM Interface - OpenAI-like Provider Examples")
    
    print("📚 Available Example Categories:\n")
    
    example_map = {}
    counter = 1
    
    for category, examples in EXAMPLES.items():
        print(f"🏷️  {category}")
        
        for module_name, description in examples.items():
            example_map[counter] = (module_name, description)
            print(f"   {counter:2d}. {description}")
            counter += 1
        
        print()  # Blank line between categories
    
    print("🔧 Special Options:")
    print(f"   {counter:2d}. Run All Examples")
    print(f"   {counter+1:2d}. Exit")
    
    return example_map


def run_example(module_name):
    """Run a specific example module."""
    try:
        print_section(f"Running Example: {module_name}")
        
        # Import the module dynamically
        module = importlib.import_module(module_name)
        
        # Check if module has a main function, otherwise run as script
        if hasattr(module, 'main'):
            module.main()
        else:
            # If no main function, the __name__ == "__main__" block should run
            # when we imported it. For safety, we'll try to find demo functions.
            demo_functions = [
                getattr(module, name) for name in dir(module) 
                if name.startswith('demo_') and callable(getattr(module, name))
            ]
            
            if demo_functions:
                print(f"Found {len(demo_functions)} demo functions:")
                for i, func in enumerate(demo_functions, 1):
                    print(f"   {i}. {func.__name__}")
                
                print()
                for func in demo_functions:
                    try:
                        print(f"🚀 Running {func.__name__}...")
                        func()
                    except Exception as e:
                        handle_error(e, f"demo function {func.__name__}")
            else:
                print("⚠️  No demo functions found in module")
        
        print(f"\n✅ Example {module_name} completed!")
        
    except ImportError as e:
        print(f"❌ Could not import {module_name}: {e}")
    except Exception as e:
        handle_error(e, f"running example {module_name}")


def run_all_examples():
    """Run all examples in sequence."""
    print_section("Running All Examples")
    
    all_modules = []
    for examples in EXAMPLES.values():
        all_modules.extend(examples.keys())
    
    print(f"📋 Will run {len(all_modules)} examples:")
    for i, module_name in enumerate(all_modules, 1):
        print(f"   {i}. {module_name}")
    
    print()
    confirm = input("Continue? (y/N): ").strip().lower()
    
    if confirm == 'y':
        for i, module_name in enumerate(all_modules, 1):
            print(f"\n{'='*60}")
            print(f"Running Example {i}/{len(all_modules)}: {module_name}")
            print(f"{'='*60}")
            
            run_example(module_name)
            
            if i < len(all_modules):
                print(f"\n⏸️  Press Enter to continue to next example...")
                input()
        
        print_section("All Examples Completed!")
    else:
        print("❌ Cancelled running all examples")


def check_dependencies():
    """Check if required dependencies are available."""
    try:
        from common import get_default_provider, get_default_executor
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Make sure you're running from the correct directory and have all required modules")
        return False


def main():
    """Main interactive loop."""
    if not check_dependencies():
        return 1
    
    while True:
        try:
            example_map = show_menu()
            max_option = len(example_map) + 2  # +2 for "Run All" and "Exit"
            
            print(f"\n👉 Enter your choice (1-{max_option}): ", end="")
            choice = input().strip()
            
            if not choice.isdigit():
                print("❌ Please enter a valid number")
                continue
            
            choice_num = int(choice)
            
            if choice_num == max_option:  # Exit
                print("👋 Goodbye!")
                break
            elif choice_num == max_option - 1:  # Run All
                run_all_examples()
            elif choice_num in example_map:
                module_name, description = example_map[choice_num]
                print(f"\n🚀 Starting: {description}")
                run_example(module_name)
            else:
                print(f"❌ Invalid choice. Please enter a number between 1 and {max_option}")
            
            if choice_num != max_option:  # Don't pause if exiting
                print(f"\n⏸️  Press Enter to return to menu...")
                input()
        
        except KeyboardInterrupt:
            print(f"\n\n🛑 Interrupted by user")
            break
        except Exception as e:
            handle_error(e, "main menu")
            print(f"\n⏸️  Press Enter to continue...")
            input()
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 