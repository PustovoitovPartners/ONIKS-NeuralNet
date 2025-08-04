"""
User-friendly CLI interface for ONIKS NeuralNet Framework.
Provides clean, intuitive interaction without technical complexity.
"""

import sys
import time
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from .runner import TaskExecutor
from .colab import is_colab_environment, is_jupyter_environment


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class ProgressBar:
    """Simple progress indicator."""
    
    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, step_description: str = ""):
        """Update progress bar."""
        self.current_step += 1
        percentage = (self.current_step / self.total_steps) * 100
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * self.current_step // self.total_steps)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        print(f"\r{Colors.CYAN}[{bar}] {percentage:5.1f}% {Colors.END}", end="")
        if step_description:
            print(f" {Colors.BLUE}{step_description}{Colors.END}", end="")
        
        if self.current_step == self.total_steps:
            print(f"\n{Colors.GREEN}✅ Completed in {elapsed:.1f}s{Colors.END}")
        
        sys.stdout.flush()


class UserFriendlyCLI:
    """Clean, user-friendly CLI interface for ONIKS."""
    
    def __init__(self):
        self.setup_logging()
        self.executor = None
        self.current_progress = None
        
    def setup_logging(self):
        """Configure logging to be less verbose for users."""
        # Only show warnings and errors to users
        logging.basicConfig(
            level=logging.WARNING,
            format='%(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        # Suppress verbose library logging
        logging.getLogger('oniks.llm').setLevel(logging.ERROR)
        logging.getLogger('oniks.agents').setLevel(logging.ERROR)
        logging.getLogger('oniks.core').setLevel(logging.ERROR)
    
    def print_header(self):
        """Print ONIKS welcome header."""
        print(f"""
{Colors.BOLD}{Colors.BLUE}╔══════════════════════════════════════════════════════════════╗
║                    🧠 ONIKS NeuralNet CLI                    ║
║              Intelligent Multi-Agent Task Execution         ║
╚══════════════════════════════════════════════════════════════╝{Colors.END}
""")
    
    def print_status(self, message: str, status: str = "info"):
        """Print status message with appropriate color."""
        colors = {
            "info": Colors.BLUE,
            "success": Colors.GREEN,
            "warning": Colors.YELLOW,
            "error": Colors.RED,
            "progress": Colors.CYAN
        }
        color = colors.get(status, Colors.BLUE)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{Colors.BOLD}[{timestamp}]{Colors.END} {color}{message}{Colors.END}")
    
    def initialize_system(self) -> bool:
        """Initialize ONIKS system with user-friendly feedback."""
        try:
            self.print_status("🚀 Initializing ONIKS system...", "progress")
            
            # Create progress callback
            def progress_callback(message: str):
                if self.current_progress:
                    self.current_progress.update(message)
                else:
                    self.print_status(f"🔄 {message}", "progress")
            
            # Initialize executor
            self.executor = TaskExecutor(progress_callback=progress_callback)
            
            self.print_status("✅ System initialized successfully", "success")
            return True
            
        except Exception as e:
            self.print_status(f"❌ System initialization failed: {str(e)}", "error")
            return False
    
    def get_user_goal(self) -> Optional[str]:
        """Get goal from user with friendly prompts."""
        # Check if we're in Colab/Jupyter environment
        if is_colab_environment() or is_jupyter_environment():
            self.print_status("⚠️  Interactive input not available in notebook environment", "warning")
            print(f"{Colors.CYAN}Use the Colab interface instead:{Colors.END}")
            print("  from oniks.ui.colab import run_task")
            print("  result = run_task('your goal here')")
            print("\nOr use the widget interface:")
            print("  from oniks.ui.colab import create_task_interface")
            print("  create_task_interface()")
            return None
        
        print(f"\n{Colors.BOLD}What would you like ONIKS to do?{Colors.END}")
        print(f"{Colors.CYAN}Examples:{Colors.END}")
        print("  • Create a Python script that calculates fibonacci numbers")
        print("  • Set up a project directory with README and config files")
        print("  • Download a file and process its contents")
        print("  • Analyze log files and create a summary report")
        print()
        
        try:
            goal = input(f"{Colors.BOLD}Your goal: {Colors.END}").strip()
            if not goal:
                self.print_status("Please enter a goal to continue", "warning")
                return None
            return goal
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Goodbye! 👋{Colors.END}")
            return None
    
    def execute_goal(self, goal: str) -> bool:
        """Execute user goal with progress tracking."""
        try:
            self.print_status(f"🎯 Goal: {goal}", "info")
            
            # Setup progress bar
            self.current_progress = ProgressBar(8, "Executing")
            
            # Execute task
            result = self.executor.execute_task(goal)
            
            # Clear progress bar reference
            self.current_progress = None
            
            if result['success']:
                self.print_status("✅ Goal completed successfully!", "success")
                self.show_results(result)
                return True
            else:
                self.print_status(f"❌ Execution failed: {result.get('error', 'Unknown error')}", "error")
                return False
            
        except Exception as e:
            self.current_progress = None
            self.print_status(f"❌ Execution failed: {str(e)}", "error")
            return False
    
    def show_results(self, result: dict):
        """Show execution results to user."""
        print(f"\n{Colors.BOLD}📊 Execution Results:{Colors.END}")
        
        # Show execution time
        execution_time = result.get('execution_time', 0)
        print(f"⏱️  Completed in {Colors.CYAN}{execution_time:.1f}s{Colors.END}")
        
        # Show created files
        created_files = result.get('created_files', [])
        for file_path in created_files:
            print(f"📁 Created file: {Colors.GREEN}{file_path}{Colors.END}")
        
        # Show executed commands
        executed_commands = result.get('executed_commands', [])
        for command in executed_commands:
            print(f"🖥️  Executed: {Colors.CYAN}{command}{Colors.END}")
        
        if not created_files and not executed_commands:
            print(f"🎉 Task completed successfully!")
        
        print()
    
    def run(self):
        """Main CLI loop."""
        self.print_header()
        
        if not self.initialize_system():
            sys.exit(1)
        
        print(f"\n{Colors.GREEN}🎉 ONIKS is ready to help!{Colors.END}")
        
        while True:
            try:
                goal = self.get_user_goal()
                if goal is None:
                    break
                
                print()  # Add spacing
                success = self.execute_goal(goal)
                
                if success:
                    print(f"\n{Colors.BOLD}Would you like to give ONIKS another task?{Colors.END}")
                    continue_choice = input(f"{Colors.CYAN}(y/n): {Colors.END}").strip().lower()
                    if continue_choice not in ['y', 'yes']:
                        break
                else:
                    print(f"\n{Colors.YELLOW}Let's try something else!{Colors.END}")
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Session ended. Goodbye! 👋{Colors.END}")
                break
            except Exception as e:
                self.print_status(f"Unexpected error: {str(e)}", "error")
                break


def main():
    """Entry point for ONIKS CLI."""
    cli = UserFriendlyCLI()
    cli.run()


if __name__ == "__main__":
    main()