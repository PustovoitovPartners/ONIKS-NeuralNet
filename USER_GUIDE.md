# 🧠 ONIKS NeuralNet - User Guide

## Quick Start

### Easy Way - Interactive CLI
```bash
python3 run_cli.py
```

### Command Line
```bash
# Interactive mode (default)
python3 -m oniks.cli run

# Demo mode  
python3 -m oniks.cli demo
```

## What ONIKS Can Do

ONIKS is an intelligent AI system that can help you with various tasks:

### 📝 **File Operations**
- Create Python scripts, config files, documentation
- Modify existing files with search and replace
- Organize project directories

### 🖥️ **System Tasks**
- Run commands and scripts
- Install software packages
- Process data files

### 🔧 **Development Tasks**
- Set up project structures
- Generate code templates
- Run tests and builds

## Example Tasks

### Simple Examples:
- "Create a Python script that prints hello world"
- "Make a directory called 'project' with a README file"
- "Create a config.json file with default settings"

### Advanced Examples:
- "Set up a Python project with main.py, requirements.txt and README"
- "Download a CSV file and create a summary report"
- "Create a backup script that archives important files"

## How It Works

1. **🎯 You describe your goal** in natural language
2. **🤖 AI creates a plan** by breaking down your goal into steps  
3. **⚡ System executes** each step automatically
4. **📊 You see the results** with clear feedback

## User Interface Features

### 🎨 **Clean Output**
- Color-coded status messages
- Progress bars for long tasks
- Clear error messages (no technical jargon)

### 📈 **Progress Tracking**  
- Real-time progress updates
- Step-by-step execution feedback
- Execution time tracking

### 🛡️ **Error Handling**
- Graceful error recovery
- User-friendly error messages  
- Automatic cleanup on failures

## Tips for Best Results

### ✅ **Good Goals:**
- Be specific about what you want
- Mention file names and formats
- Include expected outcomes

### ❌ **Avoid:**
- Vague requests like "help me with coding"
- Tasks requiring external services not available
- Destructive operations without clear intent

## Troubleshooting

### **AI Model Not Available**
If you see warnings about AI model unavailability:
1. Make sure Ollama is running on your server
2. Check that `llama3:8b` model is installed
3. System will use backup reasoning if needed

### **Permission Errors**
- Ensure you have write permissions in the current directory
- Some system operations may require elevated privileges

### **Execution Fails**
- Check that required tools (Python, etc.) are installed
- Verify file paths and permissions
- Try simpler tasks first to test the system

## Advanced Usage

### **Custom Commands**
You can ask ONIKS to:
- Install Python packages with pip
- Run git commands
- Process multiple files
- Create complex directory structures

### **Chaining Tasks**
ONIKS can handle multi-step workflows:
- "Create a Python project, write a test, and run it"
- "Download data, process it, and create a report"
- "Set up environment, install dependencies, and run tests"

## Safety Notes

- ONIKS can execute system commands - be careful with destructive operations
- Always review created files and commands before running in production
- Use in a test environment first to understand capabilities

---

**Ready to get started?** Run `python3 run_cli.py` and describe what you'd like to accomplish!