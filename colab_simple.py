"""
ONIKS для Google Colab - Простейшая версия
Скопируйте этот код в ячейку Google Colab и запустите!
"""

import subprocess
import os
import time
from pathlib import Path

def oniks_task(goal):
    """
    Главная функция ONIKS для Colab.
    Просто опишите, что хотите сделать!
    
    Примеры:
    oniks_task("Создай Python скрипт который печатает привет")
    oniks_task("Сделай калькулятор")
    oniks_task("Создай файл с текстом")
    """
    
    print("🧠 ONIKS выполняет задачу...")
    print(f"🎯 Цель: {goal}")
    print("-" * 50)
    
    goal_lower = goal.lower()
    
    # Определяем тип задачи и выполняем
    if any(word in goal_lower for word in ["привет", "hello", "print", "печат"]):
        return create_hello_script(goal)
    
    elif any(word in goal_lower for word in ["калькулятор", "calculator", "считать", "математика"]):
        return create_calculator()
    
    elif any(word in goal_lower for word in ["файл", "текст", "txt", "file"]):
        return create_text_file(goal)
    
    elif any(word in goal_lower for word in ["список", "list", "массив"]):
        return create_list_script(goal)
    
    elif any(word in goal_lower for word in ["случайн", "random", "рандом"]):
        return create_random_script(goal)
    
    else:
        return create_general_script(goal)


def create_hello_script(goal):
    """Создает скрипт Hello World"""
    filename = "hello.py"
    
    # Определяем что печатать
    if "привет" in goal.lower():
        message = "Привет от ONIKS!"
    elif "hello" in goal.lower():
        message = "Hello from ONIKS!"
    else:
        message = "Привет, мир!"
    
    code = f'''# Скрипт создан ONIKS
print("{message}")
print("Этот скрипт создан автоматически!")
'''
    
    # Создаем файл
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(code)
    
    # Запускаем
    result = subprocess.run(['python', filename], capture_output=True, text=True, encoding='utf-8')
    
    print(f"✅ Создан файл: {filename}")
    print("📄 Содержимое:")
    print(code)
    print("🖥️ Результат выполнения:")
    print(result.stdout)
    
    return {"success": True, "file": filename, "output": result.stdout}


def create_calculator():
    """Создает простой калькулятор"""
    filename = "calculator.py"
    
    code = '''# Простой калькулятор создан ONIKS

def сложить(a, b):
    return a + b

def вычесть(a, b):
    return a - b

def умножить(a, b):
    return a * b

def разделить(a, b):
    if b != 0:
        return a / b
    else:
        return "Нельзя делить на ноль!"

# Демонстрация работы
print("🧮 Калькулятор ONIKS")
print("=" * 20)
print("5 + 3 =", сложить(5, 3))
print("10 - 4 =", вычесть(10, 4))
print("6 * 7 =", умножить(6, 7))
print("15 / 3 =", разделить(15, 3))
print("10 / 0 =", разделить(10, 0))

# Интерактивное использование:
print("\\nИспользуйте функции:")
print("сложить(5, 3)")
print("вычесть(10, 4)")
print("умножить(6, 7)")
print("разделить(15, 3)")
'''
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(code)
    
    result = subprocess.run(['python', filename], capture_output=True, text=True, encoding='utf-8')
    
    print(f"✅ Создан файл: {filename}")
    print("🖥️ Результат выполнения:")
    print(result.stdout)
    
    return {"success": True, "file": filename, "output": result.stdout}


def create_text_file(goal):
    """Создает текстовый файл"""
    filename = "notes.txt"
    
    content = f"""Файл создан ONIKS
Время создания: {time.strftime('%Y-%m-%d %H:%M:%S')}
Задача: {goal}

Это текстовый файл, созданный автоматически.
Вы можете редактировать его содержимое.

Примеры использования:
- Заметки
- Списки дел  
- Идеи для проектов
- Любой текст
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Создан файл: {filename}")
    print("📄 Содержимое:")
    print(content)
    
    return {"success": True, "file": filename}


def create_list_script(goal):
    """Создает скрипт со списками"""
    filename = "lists.py"
    
    code = '''# Работа со списками - создано ONIKS

# Пример списков
fruits = ["яблоко", "банан", "апельсин", "груша", "виноград"]
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
colors = ["красный", "синий", "зеленый", "желтый", "фиолетовый"]

print("🍎 Список фруктов:")
for i, fruit in enumerate(fruits, 1):
    print(f"{i}. {fruit}")

print("\\n🔢 Числа от 1 до 10:")
print(numbers)

print("\\n🎨 Цвета:")
for color in colors:
    print(f"- {color}")

print("\\n📊 Статистика:")
print(f"Всего фруктов: {len(fruits)}")
print(f"Сумма чисел: {sum(numbers)}")
print(f"Среднее значение: {sum(numbers) / len(numbers)}")

# Добавление элементов
fruits.append("киви")
print(f"\\nДобавили киви: {fruits}")
'''
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(code)
    
    result = subprocess.run(['python', filename], capture_output=True, text=True, encoding='utf-8')
    
    print(f"✅ Создан файл: {filename}")
    print("🖥️ Результат выполнения:")
    print(result.stdout)
    
    return {"success": True, "file": filename, "output": result.stdout}


def create_random_script(goal):
    """Создает скрипт со случайными числами"""
    filename = "random_demo.py"
    
    code = '''# Генератор случайных данных - создано ONIKS
import random

print("🎲 Генератор случайных данных")
print("=" * 30)

# Случайные числа
print("🔢 Случайные числа:")
for i in range(5):
    print(f"Число {i+1}: {random.randint(1, 100)}")

# Случайный выбор
foods = ["пицца", "бургер", "суши", "паста", "салат"]
print(f"\\n🍽️ Случайное блюдо: {random.choice(foods)}")

# Случайная перестановка
cards = ["♠️", "♥️", "♦️", "♣️"]
random.shuffle(cards)
print(f"🃏 Перемешанные карты: {cards}")

# Генератор паролей
import string
password_chars = string.ascii_letters + string.digits
password = ''.join(random.choice(password_chars) for _ in range(8))
print(f"🔐 Случайный пароль: {password}")

print("\\n✨ Все данные сгенерированы случайным образом!")
'''
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(code)
    
    result = subprocess.run(['python', filename], capture_output=True, text=True, encoding='utf-8')
    
    print(f"✅ Создан файл: {filename}")
    print("🖥️ Результат выполнения:")
    print(result.stdout)
    
    return {"success": True, "file": filename, "output": result.stdout}


def create_general_script(goal):
    """Создает общий скрипт"""
    filename = "script.py"
    
    code = f'''# Скрипт создан ONIKS
# Задача: {goal}

print("🤖 Скрипт создан ONIKS")
print("📝 Задача: {goal}")
print("⏰ Время создания: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Здесь будет ваш код
print("\\n✅ Базовый шаблон готов!")
print("Отредактируйте этот файл для реализации своей задачи.")

# Пример функции
def example_function():
    return "Это пример функции"

print("🔧 Пример функции:", example_function())
'''
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(code)
    
    result = subprocess.run(['python', filename], capture_output=True, text=True, encoding='utf-8')
    
    print(f"✅ Создан файл: {filename}")
    print("🖥️ Результат выполнения:")
    print(result.stdout)
    
    return {"success": True, "file": filename, "output": result.stdout}


def показать_примеры():
    """Показывает примеры использования"""
    print("🧠 ONIKS для Google Colab")
    print("=" * 40)
    print("📚 Примеры использования:")
    print()
    print('oniks_task("Создай скрипт который печатает привет")')
    print('oniks_task("Сделай простой калькулятор")')
    print('oniks_task("Создай текстовый файл с заметками")')
    print('oniks_task("Создай скрипт со списками")')
    print('oniks_task("Создай генератор случайных чисел")')
    print()
    print("💡 Просто опишите что хотите, и ONIKS создаст это!")


def демо():
    """Запускает демонстрацию"""
    print("🎭 Демонстрация ONIKS")
    print("=" * 30)
    
    tasks = [
        "Создай скрипт который печатает привет",
        "Сделай простой калькулятор", 
        "Создай текстовый файл"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n📝 Задача {i}: {task}")
        print("-" * 25)
        oniks_task(task)
        print()


# Автоматически показываем инструкцию при импорте
print("🧠 ONIKS для Google Colab загружен!")
print("=" * 35)
print("🚀 Готов к работе!")
print()
print("📖 Попробуйте:")
print('oniks_task("Создай скрипт который печатает привет")')
print()
print("📚 Больше примеров: показать_примеры()")
print("🎭 Демонстрация: демо()")