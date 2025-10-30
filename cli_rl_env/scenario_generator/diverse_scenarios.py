"""Generate diverse scenarios that use full range of CLI commands.

This module ensures training data covers:
- All command categories (grep, sed, awk, cut, tr, etc.)
- Different file operations
- Various text processing patterns
- Git operations
- Multi-file scenarios
"""

import random
from typing import List, Dict, Any
from cli_rl_env.scenario_generator.base import (
    DifficultyLevel, FileContent, Scenario, VerificationRule
)


class DiverseScenarioGenerator:
    """Generate scenarios with diverse command usage."""
    
    # Command categories to ensure coverage
    COMMAND_CATEGORIES = {
        'file_viewing': ['cat', 'head', 'tail', 'less'],
        'file_search': ['grep', 'find'],
        'text_processing': ['sed', 'awk', 'cut', 'tr', 'sort', 'uniq'],
        'file_operations': ['cp', 'mv', 'mkdir', 'touch'],
        'text_output': ['echo', 'printf', 'tee'],
        'comparison': ['diff', 'cmp', 'comm'],
        'git': ['git'],
        'piping': ['|'],
        'redirection': ['>', '>>'],
    }
    
    def __init__(self, seed: int = None):
        """Initialize generator."""
        self.seed = seed
        if seed:
            random.seed(seed)
    
    def generate_diverse_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Generate a scenario that uses diverse commands.
        
        Args:
            difficulty: Target difficulty
            language: 'python' or 'javascript'
            
        Returns:
            Scenario with diverse command requirements
        """
        # Select scenario type based on command category
        scenario_types = [
            self._grep_intensive_scenario,
            self._sed_intensive_scenario,
            self._awk_cut_scenario,
            self._piping_scenario,
            self._multi_file_operations,
            self._git_workflow_scenario,
            self._text_transformation_scenario,
            self._file_comparison_scenario,
            self._log_analysis_scenario,
            self._refactoring_scenario,
        ]
        
        generator = random.choice(scenario_types)
        return generator(difficulty, language)
    
    def _grep_intensive_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario requiring extensive grep usage."""
        
        if language == 'python':
            main_code = '''"""Multi-module project with issues."""

# utils.py functions
def helper_one(x):
    return x + 1  # BUG: Should multiply by 2

def helper_two(x):
    return x * 2

# Main functions
def process_data(items):
    result = []
    for item in items:
        result.append(helper_one(item))  # Uses buggy function
    return result

def calculate_total(values):
    return sum(values)
'''
            
            test_code = '''"""Tests for the module."""
import pytest
from main import process_data, calculate_total

def test_process_data():
    assert process_data([1, 2, 3]) == [2, 4, 6]

def test_calculate_total():
    assert calculate_total([1, 2, 3]) == 6
'''
            
            files = [
                FileContent(path="main.py", content=main_code, is_test=False),
                FileContent(path="test_main.py", content=test_code, is_test=True),
            ]
            
            task = """The code has a bug in one of the helper functions. 
            
Use grep to:
1. Find all function definitions: grep -n "def" main.py
2. Search for the buggy helper usage: grep -r "helper_one" .
3. Check the test expectations: grep "assert" test_main.py
4. Find the bug and fix it with sed

The bug is in helper_one - it adds instead of multiplies."""
            
        else:  # JavaScript
            main_code = '''// Multi-function utility module

function helperOne(x) {
    return x + 1;  // BUG: Should multiply by 2
}

function helperTwo(x) {
    return x * 2;
}

function processData(items) {
    return items.map(helperOne);
}

module.exports = { processData, helperOne, helperTwo };
'''
            
            test_code = '''const { processData } = require('./main');

function test_processData() {
    const result = processData([1, 2, 3]);
    const expected = [2, 4, 6];
    if (JSON.stringify(result) !== JSON.stringify(expected)) {
        throw new Error(`Expected ${expected}, got ${result}`);
    }
    console.log("✓ test_processData passed");
}

test_processData();
'''
            
            files = [
                FileContent(path="main.js", content=main_code, is_test=False),
                FileContent(path="test_main.js", content=test_code, is_test=True),
            ]
            
            task = """Use grep to find and fix the bug.
            
Required commands:
1. grep -n "function" main.js
2. grep "helperOne" main.js
3. grep "Expected" test_main.js
4. Fix with sed: sed -i 's/x + 1/x * 2/g' main.js"""
        
        return Scenario(
            difficulty=difficulty,
            language=language,
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="test", target=files[1].path, 
                               description="Tests must pass")
            ],
            expected_commands=6,
            cli_history=["ls", "cat main." + ("py" if language == "python" else "js")],
            metadata={"scenario_type": "grep_intensive", "command_focus": "grep"}
        )
    
    def _sed_intensive_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario requiring multiple sed operations."""
        
        if language == 'python':
            code = '''"""Module with multiple bugs."""

def calculate(a, b):
    # BUG 1: Wrong operator
    result = a - b  # Should be a + b
    return result

def multiply(x, y):
    # BUG 2: Wrong operator
    return x + y  # Should be x * y

def divide(a, b):
    # BUG 3: Missing check
    return a / b  # Should check b != 0

# DEBUG CODE TO REMOVE
print("DEBUG: Loading module")
'''
            
            test_code = '''import pytest
from calculator import calculate, multiply, divide

def test_calculate():
    assert calculate(5, 3) == 8

def test_multiply():
    assert multiply(4, 5) == 20

def test_divide():
    assert divide(10, 2) == 5
'''
            
            files = [
                FileContent(path="calculator.py", content=code, is_test=False),
                FileContent(path="test_calculator.py", content=test_code, is_test=True),
            ]
            
            task = r"""Fix multiple bugs using sed commands.

Required:
1. sed -i 's/a - b/a + b/g' calculator.py
2. sed -i 's/x + y/x * y/g' calculator.py
3. sed -i '/DEBUG/d' calculator.py (remove debug line)
4. Add zero check: sed -i '/return a \/ b/i\    if b == 0: raise ValueError("Division by zero")' calculator.py"""
            
        else:  # JavaScript
            code = '''// Module with bugs

function calculate(a, b) {
    // BUG: Wrong operator
    return a - b;  // Should be a + b
}

function multiply(x, y) {
    // BUG: Wrong operator
    return x + y;  // Should be x * y
}

// DEBUG
console.log("DEBUG: Loading");

module.exports = { calculate, multiply };
'''
            
            test_code = '''const { calculate, multiply } = require('./calculator');

if (calculate(5, 3) !== 8) throw new Error("calculate failed");
if (multiply(4, 5) !== 20) throw new Error("multiply failed");
console.log("✓ All tests passed");
'''
            
            files = [
                FileContent(path="calculator.js", content=code, is_test=False),
                FileContent(path="test_calculator.js", content=test_code, is_test=True),
            ]
            
            task = """Fix bugs with multiple sed commands.

Use:
1. sed -i 's/a - b/a + b/g' calculator.js
2. sed -i 's/x + y/x \\* y/g' calculator.js
3. sed -i '/DEBUG/d' calculator.js"""
        
        return Scenario(
            difficulty=difficulty,
            language=language,
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="test", target=files[1].path,
                               description="Tests must pass")
            ],
            expected_commands=5,
            cli_history=["ls", "cat calculator.*"],
            metadata={"scenario_type": "sed_intensive", "command_focus": "sed"}
        )
    
    def _awk_cut_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario requiring awk/cut for text processing."""
        
        data_file = '''name,age,score
Alice,25,85
Bob,30,92
Charlie,22,78
David,35,95
'''
        
        processor = '''"""Process CSV data."""

def process_data(filename):
    # BUG: Doesn't handle header correctly
    total = 0
    with open(filename) as f:
        for line in f:
            parts = line.strip().split(',')
            total += int(parts[2])  # Will fail on header
    return total
'''
        
        test_code = '''from processor import process_data

def test_process():
    result = process_data('data.csv')
    assert result == 350, f"Expected 350, got {result}"
    print("✓ Test passed")
'''
        
        files = [
            FileContent(path="data.csv", content=data_file, is_test=False),
            FileContent(path="processor.py", content=processor, is_test=False),
            FileContent(path="test_processor.py", content=test_code, is_test=True),
        ]
        
        task = """Fix the CSV processor that fails on the header line.

Explore the data first:
1. cat data.csv
2. head -n 1 data.csv (see header)
3. tail -n +2 data.csv (see data without header)
4. cut -d',' -f3 data.csv (extract scores)
5. awk -F',' 'NR>1 {print $3}' data.csv (scores without header)

Fix by skipping header:
sed -i '/for line in f:/a\\        next(f)  # Skip header' processor.py"""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="test", target="test_processor.py",
                               description="Test must pass")
            ],
            expected_commands=8,
            cli_history=["ls", "cat data.csv | head -3"],
            metadata={"scenario_type": "awk_cut", "command_focus": "awk,cut"}
        )
    
    def _piping_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario requiring command piping."""
        
        log_file = '''2024-10-30 10:00:00 INFO Server started
2024-10-30 10:00:01 INFO Connected to database
2024-10-30 10:00:05 ERROR Failed to load config
2024-10-30 10:00:10 WARNING Retry attempt 1
2024-10-30 10:00:15 ERROR Connection timeout
2024-10-30 10:00:20 INFO Request processed
2024-10-30 10:00:25 ERROR Database query failed
'''
        
        analyzer = '''"""Log analyzer with bug."""

def count_errors(filename):
    # BUG: Counts all lines, not just errors
    count = 0
    with open(filename) as f:
        for line in f:
            count += 1  # Should filter for ERROR
    return count
'''
        
        test_code = '''from analyzer import count_errors

def test_count():
    result = count_errors('server.log')
    assert result == 3, f"Expected 3 errors, got {result}"
    print("✓ Test passed")
'''
        
        files = [
            FileContent(path="server.log", content=log_file, is_test=False),
            FileContent(path="analyzer.py", content=analyzer, is_test=False),
            FileContent(path="test_analyzer.py", content=test_code, is_test=True),
        ]
        
        task = """Fix log analyzer using piping commands.

Analyze logs with pipes:
1. cat server.log | grep ERROR (see errors)
2. cat server.log | grep ERROR | wc -l (count errors)
3. cat server.log | cut -d' ' -f4 | sort | uniq (unique levels)
4. grep ERROR server.log | cut -d' ' -f1-2 (error timestamps)

Fix the code:
sed -i 's/count += 1/if "ERROR" in line:\\n                count += 1/g' analyzer.py"""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="test", target="test_analyzer.py",
                               description="Test must pass")
            ],
            expected_commands=7,
            cli_history=["ls", "head server.log"],
            metadata={"scenario_type": "piping", "command_focus": "pipes"}
        )
    
    def _multi_file_operations(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario requiring cp, mv, mkdir operations."""
        
        utils = '''def utility_function():
    return "util"
'''
        
        main = '''from utils import utility_function

def main():
    print(utility_function())
'''
        
        files = [
            FileContent(path="utils.py", content=utils, is_test=False),
            FileContent(path="main.py", content=main, is_test=False),
        ]
        
        task = """Reorganize project structure.

Required operations:
1. mkdir lib (create directory)
2. cp utils.py lib/utils.py (copy file)
3. sed -i 's/from utils/from lib.utils/g' main.py (update import)
4. touch lib/__init__.py (make it a package)

Then verify:
5. ls lib/
6. cat main.py | grep import"""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="execution", target="main.py",
                               description="Code should run")
            ],
            expected_commands=8,
            cli_history=["ls", "tree ."],
            metadata={"scenario_type": "file_ops", "command_focus": "cp,mv,mkdir"}
        )
    
    def _git_workflow_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario involving git commands."""
        
        code = '''def feature():
    return "v1"  # Update to v2
'''
        
        files = [
            FileContent(path="feature.py", content=code, is_test=False),
        ]
        
        task = """Use git to track changes.

Workflow:
1. git init (initialize repo)
2. git add feature.py (stage file)
3. git commit -m "Initial commit" (commit)
4. sed -i 's/v1/v2/g' feature.py (make change)
5. git diff (see changes)
6. git add feature.py (stage changes)
7. git commit -m "Update to v2"
8. git log --oneline (view history)"""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="text_match", target="feature.py",
                               expected="v2", description="Version updated")
            ],
            expected_commands=10,
            cli_history=[],
            metadata={"scenario_type": "git", "command_focus": "git"}
        )
    
    def _text_transformation_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario using tr, sort, uniq."""
        
        text = '''apple
banana
apple
Cherry
banana
Apple
'''
        
        processor = '''def process():
    # BUG: Case-sensitive duplicates
    with open('words.txt') as f:
        return list(set(f.read().split()))
'''
        
        files = [
            FileContent(path="words.txt", content=text, is_test=False),
            FileContent(path="processor.py", content=processor, is_test=False),
        ]
        
        task = """Process text file using transformation commands.

Commands to explore:
1. cat words.txt | tr 'A-Z' 'a-z' (lowercase)
2. cat words.txt | tr 'A-Z' 'a-z' | sort (sorted)
3. cat words.txt | tr 'A-Z' 'a-z' | sort | uniq (unique)
4. cat words.txt | tr 'A-Z' 'a-z' | sort | uniq | wc -l (count)

Fix code to be case-insensitive:
sed -i 's/f.read().split()/f.read().lower().split()/g' processor.py"""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="text_match", target="processor.py",
                               expected="lower()", description="Uses lowercase")
            ],
            expected_commands=7,
            cli_history=["cat words.txt"],
            metadata={"scenario_type": "text_transform", "command_focus": "tr,sort,uniq"}
        )
    
    def _file_comparison_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario using diff, cmp, comm."""
        
        file1 = '''apple
banana
cherry
'''
        
        file2 = '''apple
blueberry
cherry
'''
        
        files = [
            FileContent(path="fruits1.txt", content=file1, is_test=False),
            FileContent(path="fruits2.txt", content=file2, is_test=False),
        ]
        
        task = """Compare files and create a merged version.

Commands:
1. diff fruits1.txt fruits2.txt (see differences)
2. diff -u fruits1.txt fruits2.txt (unified format)
3. comm fruits1.txt fruits2.txt (common lines)
4. cat fruits1.txt fruits2.txt | sort | uniq > merged.txt
5. cat merged.txt (verify)"""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="text_match", target="merged.txt",
                               expected="blueberry", description="Has new fruit")
            ],
            expected_commands=6,
            cli_history=["ls *.txt"],
            metadata={"scenario_type": "comparison", "command_focus": "diff,comm"}
        )
    
    def _log_analysis_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Complex log analysis requiring multiple commands."""
        
        log = '''192.168.1.1 - - [30/Oct/2024:10:00:00] "GET /api/users HTTP/1.1" 200
192.168.1.2 - - [30/Oct/2024:10:00:01] "POST /api/data HTTP/1.1" 201
192.168.1.1 - - [30/Oct/2024:10:00:05] "GET /api/users HTTP/1.1" 404
192.168.1.3 - - [30/Oct/2024:10:00:10] "GET /api/items HTTP/1.1" 200
192.168.1.2 - - [30/Oct/2024:10:00:15] "DELETE /api/data HTTP/1.1" 500
'''
        
        files = [FileContent(path="access.log", content=log, is_test=False)]
        
        task = """Analyze web server logs using various commands.

Required analysis:
1. grep "404\\|500" access.log (find errors)
2. cut -d' ' -f1 access.log | sort | uniq (unique IPs)
3. awk '{print $9}' access.log | sort | uniq -c (status code counts)
4. grep "GET" access.log | wc -l (count GET requests)
5. cat access.log | cut -d'"' -f2 | cut -d' ' -f2 (extract paths)

Create summary:
echo "Error count: $(grep -c '404\\|500' access.log)" > summary.txt"""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="text_match", target="summary.txt",
                               expected="Error count", description="Summary created")
            ],
            expected_commands=8,
            cli_history=["head -3 access.log"],
            metadata={"scenario_type": "log_analysis", "command_focus": "awk,cut,grep,pipes"}
        )
    
    def _refactoring_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Refactoring requiring find, xargs, multiple seds."""
        
        file1 = '''def old_function():
    return "old"
'''
        
        file2 = '''from module1 import old_function

def caller():
    return old_function()
'''
        
        files = [
            FileContent(path="module1.py", content=file1, is_test=False),
            FileContent(path="module2.py", content=file2, is_test=False),
        ]
        
        task = """Refactor: rename old_function to new_function everywhere.

Commands needed:
1. grep -r "old_function" . (find all occurrences)
2. find . -name "*.py" (find all Python files)
3. sed -i 's/old_function/new_function/g' module1.py
4. sed -i 's/old_function/new_function/g' module2.py
5. grep -r "new_function" . (verify changes)

Or use find with xargs:
find . -name "*.py" -exec sed -i 's/old_function/new_function/g' {} \\;"""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="text_match", target="module1.py",
                               expected="new_function", description="Renamed")
            ],
            expected_commands=6,
            cli_history=["ls *.py"],
            metadata={"scenario_type": "refactoring", "command_focus": "find,xargs,sed"}
        )

