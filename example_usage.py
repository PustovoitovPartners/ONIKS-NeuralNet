#!/usr/bin/env python3
"""
Example usage of ONIKS NeuralNet Framework with the new user-friendly CLI.
This demonstrates how to use ONIKS programmatically.
"""

from oniks.ui.runner import TaskExecutor


def example_programmatic_usage():
    """Example of using ONIKS programmatically."""
    
    print("🧠 ONIKS Programmatic Usage Example\n")
    
    # Create executor with progress callback
    def progress_callback(message: str):
        print(f"🔄 {message}")
    
    executor = TaskExecutor(progress_callback=progress_callback)
    
    # Example tasks
    tasks = [
        "Create a Python script that calculates the factorial of 5",
        "Make a simple text file with project notes",
        "Create a directory called 'examples' with a README file"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"Task {i}: {task}")
        print('='*60)
        
        # Execute task
        result = executor.execute_task(task)
        
        if result['success']:
            print(f"✅ Task completed in {result['execution_time']:.1f}s")
            
            # Show results
            if result['created_files']:
                for file_path in result['created_files']:
                    print(f"📁 Created: {file_path}")
            
            if result['executed_commands']:
                for command in result['executed_commands']:
                    print(f"🖥️ Executed: {command}")
        else:
            print(f"❌ Task failed: {result.get('error', 'Unknown error')}")


def example_interactive_cli():
    """Example of launching the interactive CLI."""
    
    print("🧠 ONIKS Interactive CLI Example\n")
    print("To use the interactive CLI, run:")
    print("  python3 run_cli.py")
    print("\nOr use the command line interface:")
    print("  python3 -m oniks.cli run")
    print("\nThe interactive CLI provides:")
    print("  • User-friendly prompts")
    print("  • Progress tracking")
    print("  • Clean error messages")
    print("  • Multi-task support")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "programmatic":
        example_programmatic_usage()
    else:
        example_interactive_cli()
        print("\nTo see programmatic usage example:")
        print("  python3 example_usage.py programmatic")