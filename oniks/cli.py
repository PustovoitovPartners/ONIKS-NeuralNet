"""Command-line interface for ONIKS NeuralNet Framework."""

import sys
import argparse
from pathlib import Path


def run_demo():
    """Run the ONIKS demonstration."""
    try:
        # Import and run the demonstration
        import importlib.util
        
        # Get the path to run_reasoning_test.py
        project_root = Path(__file__).parent.parent
        demo_script = project_root / "run_reasoning_test.py"
        
        if not demo_script.exists():
            print("Error: Demo script not found.")
            print("Please ensure run_reasoning_test.py is in the project root.")
            return 1
        
        # Load and execute the demo script
        spec = importlib.util.spec_from_file_location("demo", demo_script)
        demo_module = importlib.util.module_from_spec(spec)
        
        # Add project root to path for imports
        sys.path.insert(0, str(project_root))
        
        spec.loader.exec_module(demo_module)
        
        # Run the main function
        demo_module.main()
        return 0
        
    except Exception as e:
        print(f"Error running demo: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ONIKS NeuralNet Framework CLI",
        prog="oniks"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser(
        "demo", 
        help="Run the ONIKS demonstration"
    )
    
    args = parser.parse_args()
    
    if args.command == "demo":
        return run_demo()
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())