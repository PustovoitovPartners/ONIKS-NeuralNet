#!/usr/bin/env python3
"""
Quick start script for Google Colab users.
Run this in a Colab cell to get started with ONIKS.
"""

def colab_setup():
    """Set up ONIKS for Google Colab."""
    print("🧠 Setting up ONIKS for Google Colab...")
    
    try:
        # Try to import ONIKS
        from oniks.ui.colab import run_task, create_task_interface, quick_start, demo
        print("✅ ONIKS imported successfully!")
        
        # Show quick start guide
        quick_start()
        
        print("\n" + "="*60)
        print("🎉 ONIKS is ready! Try these commands:")
        print("="*60)
        
        print("\n1️⃣ Simple task execution:")
        print("   result = run_task('Create a Python script that prints hello')")
        
        print("\n2️⃣ Interactive widget interface:")
        print("   create_task_interface()")
        
        print("\n3️⃣ Run a demo:")
        print("   demo()")
        
        print("\n4️⃣ Example tasks to try:")
        examples = [
            "Create a Python calculator that can add and subtract numbers",
            "Make a simple data visualization with matplotlib", 
            "Create a text file with a list of Python programming tips",
            "Generate a random password generator function"
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"   • {example}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\n💡 Try installing ONIKS first:")
        print("   !pip install -e .")
        return False
    
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False


def simple_demo():
    """Run a simple demo for new users."""
    try:
        from oniks.ui.colab import run_task
        
        print("🎭 Running a simple ONIKS demo...")
        print("-" * 40)
        
        # Simple task
        task = "Create a Python script that prints 'Hello from ONIKS!'"
        print(f"Task: {task}")
        
        result = run_task(task)
        
        if result['success']:
            print("✅ Demo completed successfully!")
            print(f"⏱️  Time taken: {result['execution_time']:.1f} seconds")
            
            if result.get('created_files'):
                print("📁 Files created:")
                for file in result['created_files']:
                    print(f"   • {file}")
        else:
            print(f"❌ Demo failed: {result.get('error')}")
    
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    # Auto-run setup when executed
    success = colab_setup()
    
    if success:
        print("\n" + "="*60)
        choice = input("Would you like to run a simple demo? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            print()
            simple_demo()
    else:
        print("\n⚠️  Setup incomplete. Please resolve the issues above.")