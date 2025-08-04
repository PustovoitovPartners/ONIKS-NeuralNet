"""
Google Colab compatible interface for ONIKS NeuralNet Framework.
Provides notebook-friendly interaction with widgets and simplified APIs.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from .runner import TaskExecutor


def is_colab_environment() -> bool:
    """Detect if running in Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_jupyter_environment() -> bool:
    """Detect if running in any Jupyter environment."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


class ColabColors:
    """HTML color codes for Colab output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


class ColabONIKS:
    """Google Colab optimized interface for ONIKS."""
    
    def __init__(self):
        self.executor = None
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for Colab environment."""
        logging.basicConfig(
            level=logging.WARNING,
            format='%(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        # Suppress verbose library logging
        for logger_name in ['oniks.llm', 'oniks.agents', 'oniks.core', 'oniks.tools']:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    
    def print_header(self):
        """Print ONIKS header for Colab."""
        print(f"""
{ColabColors.BOLD}{ColabColors.BLUE}🧠 ONIKS NeuralNet Framework - Google Colab Edition{ColabColors.END}
{ColabColors.CYAN}Intelligent Multi-Agent Task Execution{ColabColors.END}
{'='*60}
""")
    
    def print_status(self, message: str, status: str = "info"):
        """Print status message with colors."""
        colors = {
            "info": ColabColors.BLUE,
            "success": ColabColors.GREEN,
            "warning": ColabColors.YELLOW,
            "error": ColabColors.RED,
            "progress": ColabColors.CYAN
        }
        color = colors.get(status, ColabColors.BLUE)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{ColabColors.BOLD}[{timestamp}]{ColabColors.END} {color}{message}{ColabColors.END}")
    
    def initialize(self) -> bool:
        """Initialize ONIKS system."""
        try:
            self.print_status("🚀 Initializing ONIKS system...", "progress")
            
            def progress_callback(message: str):
                self.print_status(f"🔄 {message}", "progress")
            
            self.executor = TaskExecutor(progress_callback=progress_callback)
            self.print_status("✅ System initialized successfully", "success")
            return True
            
        except Exception as e:
            self.print_status(f"❌ Initialization failed: {str(e)}", "error")
            return False
    
    def execute_task(self, goal: str) -> Dict[str, Any]:
        """Execute a single task."""
        if not self.executor:
            if not self.initialize():
                return {"success": False, "error": "Failed to initialize system"}
        
        self.print_status(f"🎯 Goal: {goal}", "info")
        
        try:
            result = self.executor.execute_task(goal)
            
            if result['success']:
                self.print_status("✅ Task completed successfully!", "success")
                self.show_results(result)
            else:
                self.print_status(f"❌ Task failed: {result.get('error', 'Unknown error')}", "error")
            
            return result
            
        except Exception as e:
            self.print_status(f"❌ Execution failed: {str(e)}", "error")
            return {"success": False, "error": str(e)}
    
    def show_results(self, result: Dict[str, Any]):
        """Show execution results."""
        print(f"\n{ColabColors.BOLD}📊 Execution Results:{ColabColors.END}")
        
        # Show execution time
        execution_time = result.get('execution_time', 0)
        print(f"⏱️  Completed in {ColabColors.CYAN}{execution_time:.1f}s{ColabColors.END}")
        
        # Show created files
        created_files = result.get('created_files', [])
        for file_path in created_files:
            print(f"📁 Created file: {ColabColors.GREEN}{file_path}{ColabColors.END}")
            
            # Try to show file content if it's small
            try:
                file_size = Path(file_path).stat().st_size
                if file_size < 1000:  # Show content for files < 1KB
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print(f"📄 Content preview:\n```\n{content}\n```")
            except:
                pass
        
        # Show executed commands with output
        executed_commands = result.get('executed_commands', [])
        for command in executed_commands:
            print(f"🖥️  Executed: {ColabColors.CYAN}{command}{ColabColors.END}")
        
        if not created_files and not executed_commands:
            print(f"🎉 Task completed successfully!")
        
        print()


# Global instance for easy access
oniks = ColabONIKS()


def run_task(goal: str) -> Dict[str, Any]:
    """
    Simple function to run a task with ONIKS in Colab.
    
    Usage:
        result = oniks.run_task("Create a Python script that prints hello world")
    """
    global oniks
    return oniks.execute_task(goal)


def quick_start():
    """Display quick start information for Colab users."""
    print(f"""
{ColabColors.BOLD}{ColabColors.BLUE}🚀 ONIKS Quick Start for Google Colab{ColabColors.END}

{ColabColors.BOLD}Simple Usage:{ColabColors.END}
```python
from oniks.ui.colab import run_task

# Execute a task
result = run_task("Create a Python script that calculates fibonacci numbers")
```

{ColabColors.BOLD}Advanced Usage:{ColabColors.END}
```python
from oniks.ui.colab import oniks

# Initialize system
oniks.initialize()

# Run multiple tasks
tasks = [
    "Create a Python file that prints hello world",
    "Make a simple calculator script",
    "Create a data analysis notebook template"
]

for task in tasks:
    result = oniks.execute_task(task)
    if result['success']:
        print(f"✅ Completed: {task}")
    else:
        print(f"❌ Failed: {task}")
```

{ColabColors.BOLD}Example Tasks:{ColabColors.END}
• "Create a Python script that visualizes data with matplotlib"
• "Set up a machine learning data preprocessing pipeline"  
• "Generate a CSV file with sample data and analyze it"
• "Create a web scraper that saves results to JSON"

{ColabColors.CYAN}Just describe what you want in natural language!{ColabColors.END}
""")


def demo():
    """Run a simple demonstration in Colab."""
    global oniks
    
    oniks.print_header()
    
    if not oniks.initialize():
        return
    
    print(f"{ColabColors.BOLD}Running Demo Tasks...{ColabColors.END}\n")
    
    demo_tasks = [
        "Create a Python script that prints 'Hello from ONIKS!'",
        "Make a simple text file with today's date",
        "Create a basic calculator function in Python"
    ]
    
    for i, task in enumerate(demo_tasks, 1):
        print(f"\n{ColabColors.BOLD}Demo Task {i}:{ColabColors.END} {task}")
        print("-" * 50)
        
        result = oniks.execute_task(task)
        
        if not result['success']:
            print(f"⚠️  Demo task {i} failed, continuing...")
    
    print(f"\n{ColabColors.GREEN}🎉 Demo completed! Try your own tasks with run_task().{ColabColors.END}")


# Widget-based interface for better Colab UX
def create_task_interface():
    """Create an interactive widget interface for Colab."""
    try:
        from IPython.display import display, HTML
        import ipywidgets as widgets
        
        # Task input
        task_input = widgets.Text(
            placeholder="Describe what you want ONIKS to do...",
            description="Goal:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='80%')
        )
        
        # Execute button
        execute_button = widgets.Button(
            description="Execute Task",
            button_style='primary',
            icon='play'
        )
        
        # Output area
        output = widgets.Output()
        
        def on_execute_clicked(b):
            with output:
                output.clear_output()
                if task_input.value.strip():
                    result = run_task(task_input.value.strip())
                else:
                    print("Please enter a task description.")
        
        execute_button.on_click(on_execute_clicked)
        
        # Example tasks
        examples = [
            "Create a Python script that generates random numbers",
            "Make a data visualization with matplotlib",
            "Set up a simple web scraper",
            "Create a machine learning data preprocessor"
        ]
        
        example_buttons = []
        for example in examples:
            btn = widgets.Button(
                description=example,
                layout=widgets.Layout(width='auto', margin='2px'),
                style={'button_color': '#e8f4fd'}
            )
            
            def make_example_handler(task_text):
                def handler(b):
                    task_input.value = task_text
                return handler
            
            btn.on_click(make_example_handler(example))
            example_buttons.append(btn)
        
        # Layout
        display(HTML("<h3>🧠 ONIKS Task Executor</h3>"))
        display(widgets.VBox([
            task_input,
            execute_button,
            widgets.HTML("<b>Example tasks (click to use):</b>"),
            widgets.VBox(example_buttons),
            output
        ]))
        
    except ImportError:
        print("Widgets not available. Use run_task() function instead.")
        quick_start()


# Make it easy to import everything
__all__ = [
    'ColabONIKS', 'oniks', 'run_task', 'quick_start', 'demo', 
    'create_task_interface', 'is_colab_environment', 'is_jupyter_environment'
]