#!/usr/bin/env python3
"""Test runner script for ONIKS NeuralNet framework.

This script provides convenient commands to run different test suites.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description=""):
    """Run a command and display results."""
    if description:
        print(f"\n{'='*60}")
        print(f"🧪 {description}")
        print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, cwd=Path(__file__).parent)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False


def main():
    """Main test runner function."""
    print("🚀 ONIKS NeuralNet Framework Test Runner")
    
    if len(sys.argv) < 2:
        print("""
Available test commands:
  python run_tests.py unit       - Run all unit tests
  python run_tests.py integration - Run integration tests  
  python run_tests.py core       - Run core module tests
  python run_tests.py tools      - Run tools module tests
  python run_tests.py agents     - Run agents module tests
  python run_tests.py coverage   - Run tests with coverage report
  python run_tests.py quick      - Run basic test suite
  python run_tests.py all        - Run all tests
        """)
        return
    
    test_type = sys.argv[1].lower()
    
    # Activate virtual environment command prefix
    venv_prefix = "source venv/bin/activate &&"
    
    if test_type == "unit":
        run_command(
            f"{venv_prefix} python -m pytest tests/unit/ -v",
            "Running Unit Tests"
        )
    
    elif test_type == "integration":
        run_command(
            f"{venv_prefix} python -m pytest tests/integration/ -v",
            "Running Integration Tests"
        )
    
    elif test_type == "core":
        run_command(
            f"{venv_prefix} python -m pytest tests/unit/core/ -v",
            "Running Core Module Tests"
        )
    
    elif test_type == "tools":
        run_command(
            f"{venv_prefix} python -m pytest tests/unit/tools/ -v",
            "Running Tools Module Tests"
        )
    
    elif test_type == "agents":
        run_command(
            f"{venv_prefix} python -m pytest tests/unit/agents/ -v",
            "Running Agents Module Tests"
        )
    
    elif test_type == "coverage":
        run_command(
            f"{venv_prefix} python -m pytest tests/ --cov=oniks --cov-report=html --cov-report=term",
            "Running Tests with Coverage Report"
        )
        print("\n📊 Coverage report generated in htmlcov/index.html")
    
    elif test_type == "quick":
        run_command(
            f"{venv_prefix} python -m pytest tests/unit/core/test_state.py tests/integration/test_full_system.py::TestFullSystemIntegration::test_full_system_workflow_like_run_reasoning_test -v",
            "Running Quick Test Suite"
        )
    
    elif test_type == "all":
        run_command(
            f"{venv_prefix} python -m pytest tests/ -v --tb=short",
            "Running All Tests"
        )
    
    else:
        print(f"❌ Unknown test type: {test_type}")
        print("Use 'python run_tests.py' without arguments to see available options.")


if __name__ == "__main__":
    main()