# ONIKS NeuralNet Framework

<!-- Badges Placeholders -->
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Project Status

**⚠️ Project Status: Alpha**

This is an early-stage, experimental framework. The architecture is subject to change. Not yet ready for production use.

## Overview

ONIKS is not just another LLM wrapper; it's a robust, deterministic, multi-agent framework designed for solving complex tasks through structured, tool-aware planning and execution.

Think of ONIKS as an intelligent factory rather than a single craftsman. Instead of one LLM trying to do everything, ONIKS orchestrates specialized agents working together in a coordinated, hierarchical manner. The **PlannerAgent** serves as the strategic architect, decomposing complex goals into executable steps, while the **ReasoningAgent** acts as the tactical executor, intelligently selecting and coordinating tools to accomplish each step.

This separation of concerns creates a more reliable, debuggable, and scalable approach to complex task automation, with built-in quality gates and state management ensuring robust execution even for multi-step workflows.

## Key Features

### 🏗️ Hierarchical Agency
**"Planner-Executor" architecture** separates strategic planning from tactical execution. The PlannerAgent focuses on high-level task decomposition, while the ReasoningAgent handles tool selection and execution coordination.

### 🧰 Tool-Aware Planning
The Planner generates realistic, executable plans based on the actual tools available in the system. No more abstract plans that can't be executed - every step is grounded in concrete tool capabilities.

### 🔄 Robust Orchestration
A graph-based execution engine with comprehensive state management and checkpointing ensures reliable execution. Failed steps can be recovered, and complex workflows maintain consistency across execution boundaries.

### ✅ Quality Gates
Built-in testing and verification at every stage. The framework includes comprehensive unit and integration tests, with verification mechanisms to ensure each step completes successfully before proceeding.

### 🔌 Extensible Architecture
Clean abstractions make it easy to add new tools, agents, and execution strategies. The modular design supports everything from simple file operations to complex multi-step automation workflows.

## How It Works

ONIKS follows a clear execution cycle that ensures reliable, step-by-step task completion:

```
🎯 Complex Goal
    ↓
📋 PlannerAgent (Strategic Decomposition)
    → Analyzes goal and available tools
    → Creates structured, executable plan
    ↓
🧠 ReasoningAgent (Tactical Execution)
    → Takes first task from plan
    → Selects appropriate tool
    → Coordinates execution
    ↓
🔧 Tool Execution
    → Performs concrete operations
    → Returns results and status
    ↓
🔄 Loop Back (State Management)
    → Updates state with results
    → Removes completed task from plan
    → Continues with next task
    ↓
✅ Task Complete
    → All planned tasks executed
    → Final verification and cleanup
```

This cycle continues until all tasks in the plan are completed, with built-in error handling and state checkpointing at each step.

## Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.9+ recommended)
- **Ollama** with llama3:8b model

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ONIKS_NeuralNet.git
   cd ONIKS_NeuralNet
   ```

2. **Set up the environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install and configure Ollama:**
   ```bash
   # Install Ollama (see https://ollama.ai)
   ollama serve
   ollama pull llama3:8b
   ```

### 🚀 Quick Start Options

#### 📱 **Google Colab (Easiest)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/ONIKS_NeuralNet/blob/main/ONIKS_Colab_Demo.ipynb)

```python
# In Google Colab
from oniks.ui.colab import run_task, create_task_interface

# Simple usage
result = run_task("Create a Python script that prints hello world")

# Or use interactive widgets
create_task_interface()
```

#### 💻 **Local Interactive CLI**
```bash
python3 run_cli.py
```

This launches a clean, intuitive interface where you can:
- ✅ **Give ONIKS natural language goals** (e.g., "Create a Python calculator script")
- ✅ **Track progress in real-time** with visual progress bars
- ✅ **See clear results** without technical complexity  
- ✅ **Handle multiple tasks** in one session

### 📋 Alternative: Command Line Interface

```bash
# Interactive mode (default)
python3 -m oniks.cli run

# Demo mode (technical demonstration)  
python3 -m oniks.cli demo
```

### 🔧 Advanced: Technical Demo

For developers who want to see the internal workings:

```bash
python run_reasoning_test.py
```

### Example Output

```
=== ONIKS Hierarchical PlannerAgent + ReasoningAgent Demonstration ===

1. Initializing multi-step graph and checkpoint saver...
2. Creating initial state with complex goal for hierarchical planning...
   Goal: Create a directory named 'output', and inside it, create a file 'log.txt' with the text 'System test OK'
3. Creating tools...
   Created tool: list_files
   Created tool: write_file
   Created tool: create_directory
4. Creating LLM client...
   LLM client created successfully with model 'llama3:8b'
5. Creating agents and nodes...
6. Building comprehensive graph structure...
7. Executing graph...

✅ Demonstration completed successfully!
```

## Architecture Overview

### Core Components

```python
# Agent hierarchy
from oniks.agents.planner_agent import PlannerAgent      # Strategic planning
from oniks.agents.reasoning_agent import ReasoningAgent  # Tactical execution

# Execution framework
from oniks.core.graph import Graph, ToolNode             # Graph-based orchestration
from oniks.core.state import State                       # State management
from oniks.core.checkpoint import SQLiteCheckpointSaver  # Persistence

# Tool ecosystem
from oniks.tools.fs_tools import WriteFileTool, CreateDirectoryTool
from oniks.tools.shell_tools import ExecuteBashCommandTool
from oniks.tools.core_tools import TaskCompleteTool

# LLM integration
from oniks.llm.client import OllamaClient                # Local LLM interface
```

### Framework Design Principles

1. **Separation of Concerns**: Each component has a clear, focused responsibility
2. **Tool-First Design**: All capabilities are expressed through concrete, testable tools
3. **State Transparency**: Complete state visibility and checkpoint-based recovery
4. **Fail-Safe Execution**: Comprehensive error handling and verification at each step
5. **Extensible Architecture**: Clean interfaces for adding new capabilities

## Development

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest -m unit          # Unit tests
python -m pytest -m integration   # Integration tests

# Run with coverage
python -m pytest --cov=oniks --cov-report=html
```

### Project Structure

```
oniks/
├── agents/          # Intelligent agents (Planner, Reasoning)
│   ├── base.py         # Base agent interface
│   ├── planner_agent.py        # Strategic task decomposition
│   └── reasoning_agent.py      # Tactical tool coordination
├── core/            # Core framework components
│   ├── graph.py        # Graph-based execution engine
│   ├── state.py        # State management and messaging
│   └── checkpoint.py   # Persistence and recovery
├── llm/             # LLM client integration
│   └── client.py       # Ollama client interface
└── tools/           # Tool implementations
    ├── base.py         # Tool interface
    ├── core_tools.py   # Essential framework tools
    ├── file_tools.py   # File I/O operations
    ├── fs_tools.py     # Filesystem operations
    └── shell_tools.py  # Command execution
```

## Roadmap

### Current Status (Alpha)
- ✅ Core hierarchical agent architecture
- ✅ Graph-based execution framework
- ✅ State management and checkpointing
- ✅ Comprehensive tool ecosystem
- ✅ LLM integration via Ollama
- ✅ Extensive testing infrastructure

### Upcoming Features
- 🔄 **Enhanced Error Recovery**: More sophisticated error handling and retry mechanisms
- 🔄 **Parallel Execution**: Support for concurrent task execution where possible
- 🔄 **Advanced Planning**: More sophisticated planning algorithms and optimization
- 🔄 **Tool Discovery**: Dynamic tool registration and capability discovery
- 🔄 **Monitoring & Observability**: Enhanced logging, metrics, and execution tracing

### Long-term Vision
- 🎯 **Tackling the K Prize Challenge**: Applying ONIKS to solve complex, multi-step reasoning challenges
- 🎯 **Production Readiness**: Robust deployment, scaling, and monitoring capabilities
- 🎯 **Ecosystem Growth**: Rich library of specialized tools and agents
- 🎯 **Community Building**: Active contributor community and plugin ecosystem

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Code style guidelines and standards
- Testing requirements and best practices
- Pull request process and review criteria

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Ollama](https://ollama.ai) for local LLM capabilities
- Inspired by advances in multi-agent systems and hierarchical planning
- Designed for the next generation of intelligent automation challenges

---

**Ready to build something intelligent?** Start with the [Quick Start](#quick-start) guide and explore the possibilities of hierarchical, tool-aware AI systems.