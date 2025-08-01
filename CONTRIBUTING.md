# Contributing to ONIKS NeuralNet Framework

Thank you for your interest in contributing to ONIKS! This document provides guidelines and information for contributors to help maintain code quality and project standards.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Running Tests](#running-tests)
- [Code Style Guidelines](#code-style-guidelines)
- [Contribution Process](#contribution-process)
- [Development Workflow](#development-workflow)
- [Reporting Issues](#reporting-issues)
- [Community Guidelines](#community-guidelines)

## Development Environment Setup

### Prerequisites

- **Python 3.8+** (Python 3.9+ recommended)
- **Ollama** (for LLM functionality)
- **Git** for version control

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ONIKS_NeuralNet.git
   cd ONIKS_NeuralNet
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Ollama (required for LLM functionality):**
   ```bash
   # Install Ollama (see https://ollama.ai for platform-specific instructions)
   # Start Ollama service:
   ollama serve
   
   # Pull the required model:
   ollama pull llama3:8b
   ```

5. **Verify installation:**
   ```bash
   python run_reasoning_test.py
   ```

## Running Tests

ONIKS uses pytest for testing with comprehensive unit and integration test coverage.

### Basic Test Commands

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test categories
python -m pytest -m unit          # Unit tests only
python -m pytest -m integration   # Integration tests only

# Run tests with coverage
python -m pytest --cov=oniks --cov-report=html

# Run specific test file
python -m pytest tests/unit/agents/test_planner_agent.py

# Run tests matching a pattern
python -m pytest -k "test_reasoning"
```

### Test Structure

- **Unit Tests** (`tests/unit/`): Test individual components in isolation
- **Integration Tests** (`tests/integration/`): Test component interactions and end-to-end workflows

### Test Requirements

- All new features must include comprehensive tests
- Maintain or improve code coverage
- Tests should be fast, reliable, and independent
- Use descriptive test names and docstrings

## Code Style Guidelines

ONIKS follows PEP 8 Python style guidelines with additional project-specific conventions.

### Core Principles

1. **PEP 8 Compliance**: Follow the official Python style guide
2. **Type Hints**: Use type annotations for all function parameters and return values
3. **Docstrings**: Provide comprehensive docstrings for all public classes and methods
4. **Clear Naming**: Use descriptive names for variables, functions, and classes

### Formatting Standards

```python
# Import organization
import standard_library_modules
import third_party_modules
from oniks.module import LocalImports

# Function definitions with type hints
def process_data(input_data: List[str], max_items: int = 100) -> Dict[str, Any]:
    """Process input data and return structured results.
    
    Args:
        input_data: List of strings to process.
        max_items: Maximum number of items to process.
        
    Returns:
        Dictionary containing processed results.
        
    Raises:
        ValueError: If input_data is empty.
    """
    pass

# Class definitions
class MyClass:
    """Brief description of the class purpose.
    
    More detailed description if needed.
    
    Attributes:
        attribute_name: Description of the attribute.
    """
    
    def __init__(self, param: str) -> None:
        """Initialize the class."""
        self.attribute_name = param
```

### Documentation Standards

- **Module Docstrings**: Every module should have a comprehensive docstring
- **Class Docstrings**: Include purpose, attributes, and usage examples
- **Method Docstrings**: Document parameters, return values, and exceptions
- **Type Hints**: Use for all public interfaces

### Code Quality Tools

Consider using these tools to maintain code quality:

```bash
# Code formatting (if available)
black oniks/

# Import sorting (if available)
isort oniks/

# Linting (if available)
flake8 oniks/
```

## Contribution Process

### Before Starting

1. **Check existing issues** to avoid duplicate work
2. **Create or comment on an issue** to discuss your planned contribution
3. **Fork the repository** and create a feature branch

### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the style guidelines

3. **Add tests** for new functionality

4. **Run tests** to ensure nothing is broken:
   ```bash
   python -m pytest
   ```

5. **Update documentation** if needed

### Submitting Changes

1. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "Add: Brief description of the feature"
   ```

2. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes made
   - Test results or screenshots if applicable

## Development Workflow

### Project Structure

```
oniks/
├── agents/          # Intelligent agents (Planner, Reasoning)
├── core/           # Core framework (Graph, State, Checkpointing)
├── llm/            # LLM client integration
└── tools/          # Tool implementations

tests/
├── unit/           # Unit tests
└── integration/    # Integration tests
```

### Key Components

- **Agents**: Intelligent components that make decisions (PlannerAgent, ReasoningAgent)
- **Graph**: Execution framework with nodes, edges, and state management
- **Tools**: Concrete capabilities (file operations, shell commands, etc.)
- **State**: Shared state management with checkpointing support

### Adding New Features

1. **Tools**: Extend the `Tool` base class for new capabilities
2. **Agents**: Extend `BaseAgent` for new intelligent behaviors
3. **Nodes**: Implement custom graph nodes for specialized execution logic

## Reporting Issues

### Bug Reports

Include the following information:

- **Python version** and operating system
- **Ollama version** and model being used
- **Complete error traceback**
- **Minimal reproduction case**
- **Steps to reproduce the issue**

### Feature Requests

Provide:

- **Clear description** of the proposed feature
- **Use case** and motivation
- **Possible implementation approach**
- **Impact on existing functionality**

## Community Guidelines

### Communication

- Be respectful and constructive in all interactions
- Use clear, professional language in issues and pull requests
- Provide helpful feedback during code reviews
- Ask questions if something is unclear

### Collaboration

- Give credit where due
- Be open to feedback and suggestions
- Help newcomers get started
- Share knowledge and best practices

### Quality Standards

- Write clean, maintainable code
- Test thoroughly before submitting
- Document your work clearly
- Follow established patterns and conventions

## Getting Help

- **Documentation**: Check existing docs and code examples
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Code Review**: Don't hesitate to ask for feedback

## License

By contributing to ONIKS, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to ONIKS NeuralNet Framework! Your efforts help make this project better for everyone.