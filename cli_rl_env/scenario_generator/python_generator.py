"""Generate Python code scenarios with bugs."""

import random
from typing import List

from cli_rl_env.scenario_generator.base import (
    DifficultyLevel,
    FileContent,
    Scenario,
    ScenarioGenerator,
    VerificationRule,
)
from cli_rl_env.scenario_generator.bug_injector import BugInjector
from cli_rl_env.scenario_generator.prompt_generator import PromptGenerator


class PythonScenarioGenerator(ScenarioGenerator):
    """Generate Python debugging scenarios."""
    
    def get_language(self) -> str:
        """Get the programming language."""
        return "python"
    
    def generate(self, difficulty: DifficultyLevel) -> Scenario:
        """Generate a Python scenario.
        
        Args:
            difficulty: Target difficulty level
            
        Returns:
            Complete scenario
        """
        scenario_type = random.choice(['calculator', 'data_processor', 'string_utils', 'algorithms'])
        
        if scenario_type == 'calculator':
            return self._generate_calculator_scenario(difficulty)
        elif scenario_type == 'data_processor':
            return self._generate_data_processor_scenario(difficulty)
        elif scenario_type == 'string_utils':
            return self._generate_string_utils_scenario(difficulty)
        else:
            return self._generate_algorithms_scenario(difficulty)
    
    def _generate_calculator_scenario(self, difficulty: DifficultyLevel) -> Scenario:
        """Generate a calculator scenario."""
        # Clean implementation
        main_code = '''"""Simple calculator module."""

def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

def divide(a, b):
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def power(a, b):
    """Calculate a to the power of b."""
    return a ** b
'''
        
        test_code = '''"""Tests for calculator module."""

import pytest
from calculator import add, subtract, multiply, divide, power

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 5) == -5

def test_multiply():
    assert multiply(4, 5) == 20
    assert multiply(-2, 3) == -6

def test_divide():
    assert divide(10, 2) == 5
    assert divide(7, 2) == 3.5
    with pytest.raises(ValueError):
        divide(5, 0)

def test_power():
    assert power(2, 3) == 8
    assert power(5, 0) == 1
'''
        
        # Inject bugs
        num_bugs = {
            DifficultyLevel.EASY: 1,
            DifficultyLevel.MEDIUM: 2,
            DifficultyLevel.HARD: 3,
            DifficultyLevel.VERY_HARD: 4
        }[difficulty]
        
        buggy_code, bug_descriptions = BugInjector.inject_python_bugs(
            main_code, num_bugs
        )
        
        files = [
            FileContent(path="calculator.py", content=buggy_code, is_test=False),
            FileContent(path="test_calculator.py", content=test_code, is_test=True),
        ]
        
        # Generate CLI history
        cli_history = self._generate_cli_history(difficulty, files)
        
        # Task description
        task_description = PromptGenerator.generate_debug_prompt(
            language="python",
            bug_descriptions=bug_descriptions,
            difficulty=difficulty,
            file_structure=[f.path for f in files]
        )
        
        verification_rules = [
            VerificationRule(
                type="test",
                target="test_calculator.py",
                description="All calculator tests must pass"
            ),
            VerificationRule(
                type="lint",
                target="calculator.py",
                description="Code must pass basic linting"
            )
        ]
        
        expected_commands = num_bugs * 3  # explore, identify, fix per bug
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task_description,
            files=files,
            verification_rules=verification_rules,
            expected_commands=expected_commands,
            cli_history=cli_history,
            metadata={"bugs": bug_descriptions, "scenario_type": "calculator"}
        )
    
    def _generate_data_processor_scenario(self, difficulty: DifficultyLevel) -> Scenario:
        """Generate a data processing scenario."""
        main_code = '''"""Data processing utilities."""

def filter_positive(numbers):
    """Filter positive numbers from a list."""
    return [n for n in numbers if n > 0]

def sum_even(numbers):
    """Sum all even numbers in a list."""
    return sum(n for n in numbers if n % 2 == 0)

def find_max(numbers):
    """Find maximum value in a list."""
    if not numbers:
        return None
    return max(numbers)

def average(numbers):
    """Calculate average of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def remove_duplicates(items):
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
'''
        
        test_code = '''"""Tests for data processor."""

from data_processor import filter_positive, sum_even, find_max, average, remove_duplicates

def test_filter_positive():
    assert filter_positive([1, -2, 3, -4, 5]) == [1, 3, 5]
    assert filter_positive([-1, -2, -3]) == []
    assert filter_positive([]) == []

def test_sum_even():
    assert sum_even([1, 2, 3, 4, 5, 6]) == 12
    assert sum_even([1, 3, 5]) == 0

def test_find_max():
    assert find_max([1, 5, 3, 9, 2]) == 9
    assert find_max([-5, -1, -10]) == -1
    assert find_max([]) is None

def test_average():
    assert average([1, 2, 3, 4, 5]) == 3.0
    assert average([10]) == 10.0
    assert average([]) == 0

def test_remove_duplicates():
    assert remove_duplicates([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]
    assert remove_duplicates([]) == []
'''
        
        num_bugs = {
            DifficultyLevel.EASY: 1,
            DifficultyLevel.MEDIUM: 2,
            DifficultyLevel.HARD: 3,
            DifficultyLevel.VERY_HARD: 4
        }[difficulty]
        
        buggy_code, bug_descriptions = BugInjector.inject_python_bugs(
            main_code, num_bugs
        )
        
        files = [
            FileContent(path="data_processor.py", content=buggy_code, is_test=False),
            FileContent(path="test_data_processor.py", content=test_code, is_test=True),
        ]
        
        cli_history = self._generate_cli_history(difficulty, files)
        
        task_description = PromptGenerator.generate_debug_prompt(
            language="python",
            bug_descriptions=bug_descriptions,
            difficulty=difficulty,
            file_structure=[f.path for f in files]
        )
        
        verification_rules = [
            VerificationRule(
                type="test",
                target="test_data_processor.py",
                description="All data processor tests must pass"
            )
        ]
        
        expected_commands = num_bugs * 3
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task_description,
            files=files,
            verification_rules=verification_rules,
            expected_commands=expected_commands,
            cli_history=cli_history,
            metadata={"bugs": bug_descriptions, "scenario_type": "data_processor"}
        )
    
    def _generate_string_utils_scenario(self, difficulty: DifficultyLevel) -> Scenario:
        """Generate a string utilities scenario."""
        main_code = '''"""String utility functions."""

def reverse_string(s):
    """Reverse a string."""
    return s[::-1]

def is_palindrome(s):
    """Check if string is a palindrome."""
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]

def count_vowels(s):
    """Count vowels in a string."""
    vowels = "aeiouAEIOU"
    return sum(1 for char in s if char in vowels)

def capitalize_words(s):
    """Capitalize first letter of each word."""
    return " ".join(word.capitalize() for word in s.split())

def remove_whitespace(s):
    """Remove all whitespace from string."""
    return "".join(s.split())
'''
        
        test_code = '''"""Tests for string utilities."""

from string_utils import reverse_string, is_palindrome, count_vowels, capitalize_words, remove_whitespace

def test_reverse_string():
    assert reverse_string("hello") == "olleh"
    assert reverse_string("") == ""
    assert reverse_string("a") == "a"

def test_is_palindrome():
    assert is_palindrome("racecar") == True
    assert is_palindrome("hello") == False
    assert is_palindrome("A man a plan a canal Panama") == True

def test_count_vowels():
    assert count_vowels("hello") == 2
    assert count_vowels("AEIOU") == 5
    assert count_vowels("xyz") == 0

def test_capitalize_words():
    assert capitalize_words("hello world") == "Hello World"
    assert capitalize_words("python programming") == "Python Programming"

def test_remove_whitespace():
    assert remove_whitespace("hello world") == "helloworld"
    assert remove_whitespace("  a  b  c  ") == "abc"
'''
        
        num_bugs = {
            DifficultyLevel.EASY: 1,
            DifficultyLevel.MEDIUM: 2,
            DifficultyLevel.HARD: 3,
            DifficultyLevel.VERY_HARD: 4
        }[difficulty]
        
        buggy_code, bug_descriptions = BugInjector.inject_python_bugs(
            main_code, num_bugs
        )
        
        files = [
            FileContent(path="string_utils.py", content=buggy_code, is_test=False),
            FileContent(path="test_string_utils.py", content=test_code, is_test=True),
        ]
        
        cli_history = self._generate_cli_history(difficulty, files)
        
        task_description = PromptGenerator.generate_debug_prompt(
            language="python",
            bug_descriptions=bug_descriptions,
            difficulty=difficulty,
            file_structure=[f.path for f in files]
        )
        
        verification_rules = [
            VerificationRule(
                type="test",
                target="test_string_utils.py",
                description="All string utility tests must pass"
            )
        ]
        
        expected_commands = num_bugs * 3
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task_description,
            files=files,
            verification_rules=verification_rules,
            expected_commands=expected_commands,
            cli_history=cli_history,
            metadata={"bugs": bug_descriptions, "scenario_type": "string_utils"}
        )
    
    def _generate_algorithms_scenario(self, difficulty: DifficultyLevel) -> Scenario:
        """Generate an algorithms scenario."""
        main_code = '''"""Algorithm implementations."""

def binary_search(arr, target):
    """Binary search in sorted array."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def bubble_sort(arr):
    """Sort array using bubble sort."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

def factorial(n):
    """Calculate factorial of n."""
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
'''
        
        test_code = '''"""Tests for algorithms."""

from algorithms import binary_search, bubble_sort, fibonacci, factorial

def test_binary_search():
    assert binary_search([1, 2, 3, 4, 5], 3) == 2
    assert binary_search([1, 2, 3, 4, 5], 6) == -1
    assert binary_search([1], 1) == 0

def test_bubble_sort():
    assert bubble_sort([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]
    assert bubble_sort([]) == []
    assert bubble_sort([1]) == [1]

def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5
    assert fibonacci(10) == 55

def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120
'''
        
        num_bugs = {
            DifficultyLevel.EASY: 1,
            DifficultyLevel.MEDIUM: 2,
            DifficultyLevel.HARD: 3,
            DifficultyLevel.VERY_HARD: 4
        }[difficulty]
        
        buggy_code, bug_descriptions = BugInjector.inject_python_bugs(
            main_code, num_bugs
        )
        
        files = [
            FileContent(path="algorithms.py", content=buggy_code, is_test=False),
            FileContent(path="test_algorithms.py", content=test_code, is_test=True),
        ]
        
        cli_history = self._generate_cli_history(difficulty, files)
        
        task_description = PromptGenerator.generate_debug_prompt(
            language="python",
            bug_descriptions=bug_descriptions,
            difficulty=difficulty,
            file_structure=[f.path for f in files]
        )
        
        verification_rules = [
            VerificationRule(
                type="test",
                target="test_algorithms.py",
                description="All algorithm tests must pass"
            )
        ]
        
        expected_commands = num_bugs * 3
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task_description,
            files=files,
            verification_rules=verification_rules,
            expected_commands=expected_commands,
            cli_history=cli_history,
            metadata={"bugs": bug_descriptions, "scenario_type": "algorithms"}
        )
    
    def _generate_cli_history(self, difficulty: DifficultyLevel, files: List[FileContent]) -> List[str]:
        """Generate simulated CLI history based on difficulty."""
        history = []
        
        if difficulty == DifficultyLevel.EASY:
            # Minimal history for easy tasks
            history.append("$ ls")
            history.append(" ".join([f.path for f in files]))
        else:
            # More detailed history for harder tasks
            history.append("$ ls -la")
            for f in files:
                history.append(f"-rw-r--r-- 1 user user {len(f.content)} Oct 30 10:00 {f.path}")
            
            if difficulty in [DifficultyLevel.HARD, DifficultyLevel.VERY_HARD]:
                # Add some test output
                history.append("$ pytest -v")
                history.append("test_*.py::test_* FAILED")
                history.append("Some tests are failing...")
        
        return history

