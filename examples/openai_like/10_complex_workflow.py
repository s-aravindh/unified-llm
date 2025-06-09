#!/usr/bin/env python3
"""
Example: Complex Workflow
Demonstrates: Advanced interaction patterns and workflow orchestration

This example shows:
- Multi-step problem solving with tools
- Workflow state management
- Error handling and recovery
- Result aggregation and reporting
"""

import time
from common import (
    get_default_provider, get_default_executor, 
    calculate, get_weather, search_web,
    print_section, print_subsection, handle_error, execute_tool_calls
)


def demo_data_analysis_workflow():
    """Demonstrate a data analysis workflow."""
    
    print_section("Data Analysis Workflow")
    
    try:
        provider = get_default_provider(tools=[calculate])
        executor = get_default_executor()
        
        print("🤖 Starting data analysis workflow...")
        
        # Workflow state
        workflow_state = {
            'data_points': [23, 45, 67, 12, 89, 34, 56, 78, 91, 25],
            'calculations': {},
            'insights': []
        }
        
        messages = []
        
        # Step 1: Calculate basic statistics
        print_subsection("Step 1: Basic Statistics")
        
        data_str = str(workflow_state['data_points'])
        user_input = f"I have this dataset: {data_str}. Please calculate the sum of all values."
        messages.append({"role": "user", "content": user_input})
        
        response = provider.chat(messages)
        print(f"🤖 Assistant: {response.content}")
        
        if response.tool_calls:
            messages.append({"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})
            tool_results = execute_tool_calls(executor, response.tool_calls)
            messages.extend(tool_results)
            
            # Get the sum from tool results
            for result in tool_results:
                if 'content' in result:
                    try:
                        workflow_state['calculations']['sum'] = float(result['content'])
                        break
                    except:
                        pass
            
            final_response = provider.chat(messages)
            print(f"🤖 Assistant (final): {final_response.content}")
            messages.append({"role": "assistant", "content": final_response.content})
        
        # Step 2: Calculate average
        print_subsection("Step 2: Average Calculation")
        
        if 'sum' in workflow_state['calculations']:
            count = len(workflow_state['data_points'])
            user_input = f"Now calculate the average by dividing {workflow_state['calculations']['sum']} by {count}."
            messages.append({"role": "user", "content": user_input})
            
            response = provider.chat(messages)
            print(f"🤖 Assistant: {response.content}")
            
            if response.tool_calls:
                messages.append({"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})
                tool_results = execute_tool_calls(executor, response.tool_calls)
                messages.extend(tool_results)
                
                # Store average
                for result in tool_results:
                    if 'content' in result:
                        try:
                            workflow_state['calculations']['average'] = float(result['content'])
                            break
                        except:
                            pass
                
                final_response = provider.chat(messages)
                messages.append({"role": "assistant", "content": final_response.content})
        
        # Step 3: Analysis and insights
        print_subsection("Step 3: Insights Generation")
        
        user_input = f"Based on our calculations - sum: {workflow_state['calculations'].get('sum', 'unknown')}, average: {workflow_state['calculations'].get('average', 'unknown')} - what insights can you provide about this dataset?"
        messages.append({"role": "user", "content": user_input})
        
        response = provider.chat(messages)
        print(f"🤖 Assistant: {response.content}")
        workflow_state['insights'].append(response.content)
        
        # Step 4: Workflow summary
        print_subsection("Workflow Summary")
        
        print(f"📊 Workflow Results:")
        print(f"   Dataset size: {len(workflow_state['data_points'])} points")
        print(f"   Sum: {workflow_state['calculations'].get('sum', 'Not calculated')}")
        print(f"   Average: {workflow_state['calculations'].get('average', 'Not calculated')}")
        print(f"   Insights generated: {len(workflow_state['insights'])}")
        print(f"   Total conversation turns: {len([m for m in messages if m['role'] == 'user'])}")
        
        provider.close()
        
        print("\n✅ Data analysis workflow completed!")
        
    except Exception as e:
        handle_error(e, "data analysis workflow")


def demo_problem_solving_pipeline():
    """Demonstrate a complex problem-solving pipeline."""
    
    print_section("Problem Solving Pipeline")
    
    try:
        provider = get_default_provider(tools=[calculate, get_weather, search_web])
        executor = get_default_executor()
        
        print("🤖 Starting problem-solving pipeline...")
        
        # Complex scenario: Planning a outdoor event
        scenario = {
            'event': 'Outdoor concert',
            'expected_attendees': 500,
            'budget_per_person': 25,
            'location': 'San Francisco',
            'steps_completed': [],
            'decisions': {}
        }
        
        messages = []
        
        def execute_pipeline_step(step_name, user_input):
            """Execute a single pipeline step."""
            print_subsection(f"Pipeline Step: {step_name}")
            
            messages.append({"role": "user", "content": user_input})
            
            response = provider.chat(messages)
            print(f"🤖 Assistant: {response.content}")
            
            # Handle tool calls
            if response.tool_calls:
                messages.append({"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})
                tool_results = execute_tool_calls(executor, response.tool_calls)
                messages.extend(tool_results)
                
                final_response = provider.chat(messages)
                print(f"🤖 Assistant (after tools): {final_response.content}")
                messages.append({"role": "assistant", "content": final_response.content})
            else:
                messages.append({"role": "assistant", "content": response.content})
            
            scenario['steps_completed'].append(step_name)
            return response.content
        
        # Step 1: Budget calculation
        budget_result = execute_pipeline_step(
            "Budget Calculation",
            f"Calculate the total budget for an outdoor concert with {scenario['expected_attendees']} attendees at ${scenario['budget_per_person']} per person."
        )
        
        # Step 2: Weather check
        weather_result = execute_pipeline_step(
            "Weather Assessment",
            f"Check the weather in {scenario['location']} for planning purposes."
        )
        
        # Step 3: Research requirements
        research_result = execute_pipeline_step(
            "Requirements Research", 
            "Search for information about outdoor event permits and requirements."
        )
        
        # Step 4: Final recommendations
        final_result = execute_pipeline_step(
            "Final Recommendations",
            f"Based on our budget calculation, weather check, and requirements research, provide final recommendations for the {scenario['event']} in {scenario['location']}."
        )
        
        # Pipeline results
        print_subsection("Pipeline Results")
        
        print(f"📊 Problem-Solving Pipeline Summary:")
        print(f"   Scenario: {scenario['event']} for {scenario['expected_attendees']} people")
        print(f"   Location: {scenario['location']}")
        print(f"   Steps completed: {len(scenario['steps_completed'])}")
        
        for i, step in enumerate(scenario['steps_completed'], 1):
            print(f"   {i}. {step} ✅")
        
        print(f"   Total messages: {len(messages)}")
        print(f"   Tool executions: {len([m for m in messages if m.get('role') == 'tool'])}")
        
        provider.close()
        
        print("\n✅ Problem-solving pipeline completed!")
        
    except Exception as e:
        handle_error(e, "problem solving pipeline")


def demo_error_recovery_workflow():
    """Demonstrate error recovery in workflows."""
    
    print_section("Error Recovery Workflow")
    
    try:
        provider = get_default_provider(tools=[calculate])
        executor = get_default_executor()
        
        print("🤖 Starting error recovery workflow...")
        
        messages = []
        workflow_attempts = []
        
        def attempt_calculation(expression, attempt_number):
            """Attempt a calculation with error handling."""
            print_subsection(f"Attempt {attempt_number}")
            
            user_input = f"Calculate: {expression}"
            messages.append({"role": "user", "content": user_input})
            
            try:
                response = provider.chat(messages)
                print(f"🤖 Assistant: {response.content}")
                
                if response.tool_calls:
                    messages.append({"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})
                    
                    # Try to execute tools
                    try:
                        tool_results = execute_tool_calls(executor, response.tool_calls)
                        messages.extend(tool_results)
                        
                        final_response = provider.chat(messages)
                        print(f"🤖 Assistant (final): {final_response.content}")
                        messages.append({"role": "assistant", "content": final_response.content})
                        
                        workflow_attempts.append({
                            'attempt': attempt_number,
                            'expression': expression,
                            'status': 'success',
                            'result': tool_results[0]['content'] if tool_results else 'unknown'
                        })
                        
                        return True, tool_results[0]['content'] if tool_results else None
                        
                    except Exception as tool_error:
                        print(f"❌ Tool execution failed: {tool_error}")
                        
                        # Add error context to conversation
                        error_message = f"The calculation failed with error: {tool_error}. Please suggest an alternative approach."
                        messages.append({"role": "user", "content": error_message})
                        
                        recovery_response = provider.chat(messages)
                        print(f"🔄 Recovery suggestion: {recovery_response.content}")
                        messages.append({"role": "assistant", "content": recovery_response.content})
                        
                        workflow_attempts.append({
                            'attempt': attempt_number,
                            'expression': expression,
                            'status': 'tool_error',
                            'error': str(tool_error)
                        })
                        
                        return False, None
                else:
                    # No tool calls, just conversational response
                    messages.append({"role": "assistant", "content": response.content})
                    
                    workflow_attempts.append({
                        'attempt': attempt_number,
                        'expression': expression,
                        'status': 'no_tools',
                        'response': response.content
                    })
                    
                    return True, "No calculation performed"
                    
            except Exception as e:
                print(f"❌ Request failed: {e}")
                
                workflow_attempts.append({
                    'attempt': attempt_number,
                    'expression': expression,
                    'status': 'request_error',
                    'error': str(e)
                })
                
                return False, None
        
        # Test various calculations with potential errors
        calculations = [
            "25 * 4",  # Should work
            "10 / 2",  # Should work
            "invalid expression!@#",  # Might cause error
            "100 - 25"  # Recovery attempt
        ]
        
        for i, calc in enumerate(calculations, 1):
            success, result = attempt_calculation(calc, i)
            
            if success:
                print(f"✅ Attempt {i} succeeded: {result}")
            else:
                print(f"❌ Attempt {i} failed, continuing...")
            
            time.sleep(0.5)  # Brief pause between attempts
        
        # Workflow summary
        print_subsection("Error Recovery Summary")
        
        print(f"📊 Workflow Execution Summary:")
        
        successful_attempts = [a for a in workflow_attempts if a['status'] == 'success']
        failed_attempts = [a for a in workflow_attempts if a['status'] in ['tool_error', 'request_error']]
        
        print(f"   Total attempts: {len(workflow_attempts)}")
        print(f"   Successful: {len(successful_attempts)}")
        print(f"   Failed: {len(failed_attempts)}")
        print(f"   Success rate: {len(successful_attempts)/len(workflow_attempts)*100:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for attempt in workflow_attempts:
            status_emoji = "✅" if attempt['status'] == 'success' else "❌"
            print(f"   {status_emoji} Attempt {attempt['attempt']}: {attempt['expression']} -> {attempt['status']}")
        
        provider.close()
        
        print("\n✅ Error recovery workflow completed!")
        
    except Exception as e:
        handle_error(e, "error recovery workflow")


def demo_streaming_workflow():
    """Demonstrate complex workflow with streaming."""
    
    print_section("Streaming Complex Workflow")
    
    try:
        provider = get_default_provider(tools=[calculate, get_weather])
        executor = get_default_executor()
        
        print("🤖 Starting streaming complex workflow...")
        
        messages = []
        workflow_steps = [
            "Calculate the area of a circle with radius 5",
            "Check the weather in New York",
            "Based on the circle area and weather, make a recommendation for outdoor activities"
        ]
        
        def streaming_workflow_step(step_description, step_number):
            """Execute a workflow step with streaming."""
            print_subsection(f"Streaming Step {step_number}")
            
            messages.append({"role": "user", "content": step_description})
            
            print(f"👤 Request: {step_description}")
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
            
            print()  # New line
            
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
                
                # Stream continuation
                print("🤖 Assistant (continued): ", end="", flush=True)
                
                for chunk in provider.stream(messages):
                    if chunk.content_delta:
                        print(chunk.content_delta, end="", flush=True)
                    if chunk.finish_reason:
                        break
                
                print()  # New line
            else:
                messages.append({"role": "assistant", "content": full_content})
        
        # Execute streaming workflow
        for i, step in enumerate(workflow_steps, 1):
            streaming_workflow_step(step, i)
            time.sleep(0.5)  # Brief pause between steps
        
        print_subsection("Streaming Workflow Complete")
        
        print(f"📊 Streaming Workflow Summary:")
        print(f"   Steps executed: {len(workflow_steps)}")
        print(f"   Total messages: {len(messages)}")
        print(f"   Tool messages: {len([m for m in messages if m.get('role') == 'tool'])}")
        
        provider.close()
        
        print("\n✅ Streaming complex workflow completed!")
        
    except Exception as e:
        handle_error(e, "streaming workflow")


if __name__ == "__main__":
    demo_data_analysis_workflow()
    demo_problem_solving_pipeline()
    demo_error_recovery_workflow()
    demo_streaming_workflow() 