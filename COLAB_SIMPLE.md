# 🧠 ONIKS для Google Colab - Простейший способ

## 🚀 Как запустить (3 шага):

### Шаг 1: Откройте Google Colab
Идите на [colab.research.google.com](https://colab.research.google.com) и создайте новый notebook

### Шаг 2: Скопируйте код
Скопируйте весь код из файла `colab_simple.py` и вставьте в первую ячейку Colab

### Шаг 3: Запустите и используйте!
```python
# Запустите ячейку (Ctrl+Enter или кнопка ▶️)
# После запуска можете писать команды:

oniks_task("Создай скрипт который печатает привет")
```

## 📱 Готовый код для копирования:

```python
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

# ... (остальной код из colab_simple.py)
```

## 🎯 Примеры использования:

После запуска кода, в новых ячейках пишите:

```python
# Создать скрипт Hello World
oniks_task("Создай скрипт который печатает привет")
```

```python
# Создать калькулятор
oniks_task("Сделай простой калькулятор")
```

```python
# Создать текстовый файл
oniks_task("Создай текстовый файл с заметками")
```

```python
# Создать скрипт со списками
oniks_task("Создай скрипт со списками")
```

```python
# Создать генератор случайных чисел
oniks_task("Создай генератор случайных чисел")
```

```python
# Показать все примеры
показать_примеры()
```

```python
# Запустить демонстрацию
демо()
```

## 🎬 Что вы увидите:

После выполнения команды `oniks_task("Создай скрипт который печатает привет")`:

```
🧠 ONIKS выполняет задачу...
🎯 Цель: Создай скрипт который печатает привет
--------------------------------------------------
✅ Создан файл: hello.py
📄 Содержимое:
# Скрипт создан ONIKS
print("Привет от ONIKS!")
print("Этот скрипт создан автоматически!")

🖥️ Результат выполнения:
Привет от ONIKS!
Этот скрипт создан автоматически!
```

## 💡 Преимущества этого способа:

- ✅ **Один файл** - просто скопировал и работает
- ✅ **Никаких установок** - не нужен pip install
- ✅ **Простые команды** - пишите на русском или английском
- ✅ **Мгновенный результат** - файлы создаются и запускаются сразу
- ✅ **Работает везде** - в любом Google Colab notebook

## 🔥 Готовая ссылка для пользователей:

**"Хотите попробовать ONIKS в Google Colab? Скопируйте код из файла `colab_simple.py` в новый Colab notebook и запустите!"**

Это максимально простой способ - один файл, никаких сложностей! 🚀