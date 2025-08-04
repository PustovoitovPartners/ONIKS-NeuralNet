# 🧠 ONIKS for Google Colab

## 🚀 Quick Start

### 1. Open the Demo Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/ONIKS_NeuralNet/blob/main/ONIKS_Colab_Demo.ipynb)

### 2. Simple Usage

```python
# Install (if needed)
!pip install -e .

# Import and use
from oniks.ui.colab import run_task

# Execute a task
result = run_task("Create a Python script that prints hello world")
```

### 3. Widget Interface (Recommended)

```python
from oniks.ui.colab import create_task_interface

# Creates an interactive interface with buttons and text input
create_task_interface()
```

## 📱 Colab-Specific Features

### ✅ **What Works in Colab:**
- 🎯 **Simple task execution** with `run_task()`
- 🎨 **Interactive widgets** with `create_task_interface()`
- 📊 **Progress tracking** with visual indicators
- 🔄 **Multiple task execution** in sequence
- 📁 **File content preview** for small files
- 🎭 **Built-in demo** with `demo()`

### 🚫 **Colab Limitations:**
- No interactive terminal input (use functions instead)
- Limited file system access (Colab sandbox)
- Some system commands may be restricted

## 🎯 Example Tasks Perfect for Colab

### **Data Science & Analysis:**
```python
run_task("Create a Python script that generates sample data and plots it with matplotlib")
run_task("Make a data preprocessing function for machine learning")
run_task("Create a simple linear regression example with visualization")
```

### **Utilities & Tools:**
```python
run_task("Create a function that calculates statistics from a list of numbers")
run_task("Make a password generator with customizable length and characters")
run_task("Create a text file parser that extracts specific information")
```

### **Learning & Examples:**
```python
run_task("Create examples of Python list comprehensions with explanations")
run_task("Make a simple class demonstration with methods and properties")
run_task("Create a function that demonstrates different sorting algorithms")
```

## 🎨 Interactive Widget Interface

The widget interface provides:
- 📝 Text input field for task descriptions
- ▶️ Execute button to run tasks
- 🎯 Example task buttons (click to use)
- 📊 Real-time output display
- 🔄 Multiple task execution support

```python
from oniks.ui.colab import create_task_interface
create_task_interface()
```

## 🔧 Advanced Usage

### Multiple Tasks
```python
from oniks.ui.colab import oniks

# Initialize once
oniks.initialize()

# Run multiple tasks
tasks = [
    "Create a CSV file with sample customer data",
    "Create a Python script to read and analyze the CSV",
    "Generate a summary report of the analysis"
]

for task in tasks:
    result = oniks.execute_task(task)
    print(f"✅ Completed: {task}" if result['success'] else f"❌ Failed: {task}")
```

### Custom Progress Tracking
```python
from oniks.ui.runner import TaskExecutor

def my_progress_callback(message):
    print(f"🔄 {message}")

executor = TaskExecutor(progress_callback=my_progress_callback)
result = executor.execute_task("Your task here")
```

## 🆘 Troubleshooting

### **Common Issues:**

1. **Import Errors:**
   ```python
   # Make sure ONIKS is properly installed
   !pip install -e .
   ```

2. **Permission Errors:**
   ```python
   # Some system operations may be limited in Colab
   # Try file-focused tasks instead
   ```

3. **Widget Not Displaying:**
   ```python
   # Make sure ipywidgets is available
   !pip install ipywidgets
   ```

### **Best Practices:**

- ✅ Start with simple tasks to test the system
- ✅ Use specific, clear task descriptions
- ✅ Check created files with `!ls` or `%ls`
- ✅ Use the widget interface for better UX

## 📚 Learning Resources

### **Example Notebooks:**
- [Basic Usage](ONIKS_Colab_Demo.ipynb) - Getting started with ONIKS
- [Data Science Examples] - Common data science tasks
- [Automation Examples] - File and system automation

### **Task Templates:**

**File Operations:**
- "Create a [type] file with [content]"
- "Read file [name] and [action]"
- "Process all [type] files in current directory"

**Code Generation:**
- "Create a Python [type] that [function]"
- "Make a function that [action] and returns [result]"
- "Generate example code for [concept]"

**Data Processing:**
- "Create sample data for [domain] and save as [format]"
- "Analyze [type] data and create [visualization]"
- "Process [format] file and extract [information]"

## 🎉 Ready to Start?

1. **Open the demo notebook** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/ONIKS_NeuralNet/blob/main/ONIKS_Colab_Demo.ipynb)

2. **Try the widget interface:**
   ```python
   from oniks.ui.colab import create_task_interface
   create_task_interface()
   ```

3. **Start with simple tasks:**
   ```python
   from oniks.ui.colab import run_task
   result = run_task("Create a Python script that prints your name")
   ```

**Happy coding with ONIKS in Google Colab! 🚀**