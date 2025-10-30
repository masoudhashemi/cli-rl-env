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
            self._archive_compression_scenario,
            self._batch_processing_scenario,
            self._complex_redirection_scenario,
            self._symbolic_links_scenario,
            self._permissions_scenario,
            self._data_pipeline_scenario,
            self._config_editing_scenario,
            self._directory_tree_scenario,
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
            
            task = """The code has a bug in one of the helper functions. The tests are failing. You need to explore the codebase, identify which function is buggy, understand what it should do based on the test expectations, and fix it."""
            
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
            
            task = """The JavaScript code has a bug that's causing test failures. Search through the code to find the issue and fix it."""
        
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
            metadata={
                "scenario_type": "grep_intensive",
                "command_focus": "grep",
                "solution_steps": [
                    "Find all function definitions: grep -n 'def' main.py (or 'function' for JS)",
                    "Search for the buggy helper usage: grep -r 'helper_one' .",
                    "Check the test expectations: grep 'assert' test file",
                    "Identify that helper_one adds instead of multiplies",
                    "Fix with sed: sed -i 's/x + 1/x * 2/g' main file",
                    "Run tests to verify fix"
                ]
            }
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
            
            task = """The calculator module has multiple bugs that are causing test failures. There are also debug statements that should be removed. Find and fix all issues to make the tests pass."""
            
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
            
            task = """The calculator module has bugs causing test failures. There are also debug statements that need to be removed. Fix all the issues."""
        
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
            metadata={
                "scenario_type": "sed_intensive",
                "command_focus": "sed",
                "solution_steps": [
                    "Fix subtract to add: sed -i 's/a - b/a + b/g' calculator file",
                    "Fix add to multiply: sed -i 's/x + y/x * y/g' calculator file",
                    "Remove debug lines: sed -i '/DEBUG/d' calculator file",
                    "Add zero check for divide (Python): sed -i '/return a / b/i\\    if b == 0: raise ValueError(...)' calculator.py",
                    "Run tests to verify all fixes"
                ]
            }
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
        
        task = """The CSV processor is failing tests. The program processes a CSV file but seems to have an issue with how it reads the data. Investigate the data file structure and fix the processor to handle it correctly."""
        
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
            metadata={
                "scenario_type": "awk_cut",
                "command_focus": "awk,cut",
                "solution_steps": [
                    "Examine data file: cat data.csv",
                    "See header: head -n 1 data.csv",
                    "See data without header: tail -n +2 data.csv",
                    "Extract scores column: cut -d',' -f3 data.csv",
                    "Extract scores without header: awk -F',' 'NR>1 {print $3}' data.csv",
                    "Identify bug: processor doesn't skip header line",
                    "Fix by adding skip: sed -i '/for line in f:/a\\        next(f)  # Skip header' processor.py",
                    "Run test to verify"
                ]
            }
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
        
        task = """The log analyzer is failing tests. It should count error messages in the log file, but it's returning the wrong value. Examine the log file and the code to understand what's wrong and fix it."""
        
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
            metadata={
                "scenario_type": "piping",
                "command_focus": "pipes",
                "solution_steps": [
                    "View log: cat server.log",
                    "Filter errors: cat server.log | grep ERROR",
                    "Count errors: cat server.log | grep ERROR | wc -l",
                    "Check log levels: cat server.log | cut -d' ' -f4 | sort | uniq",
                    "Identify bug: code counts all lines instead of ERROR lines",
                    "Fix: sed -i 's/count += 1/if \"ERROR\" in line:\\n                count += 1/g' analyzer.py",
                    "Run test to verify"
                ]
            }
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
        
        task = """Reorganize the project structure by moving utils.py into a new 'lib' directory and updating imports accordingly. Make sure the code still runs after the reorganization."""
        
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
            metadata={
                "scenario_type": "file_ops",
                "command_focus": "cp,mv,mkdir",
                "solution_steps": [
                    "Create directory: mkdir lib",
                    "Copy file: cp utils.py lib/utils.py",
                    "Update import: sed -i 's/from utils/from lib.utils/g' main.py",
                    "Make package: touch lib/__init__.py",
                    "Verify structure: ls lib/",
                    "Check import: cat main.py | grep import",
                    "Test execution: python main.py"
                ]
            }
        )
    
    def _git_workflow_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario involving git commands."""
        
        code = '''def feature():
    return "v1"  # Update to v2
'''
        
        files = [
            FileContent(path="feature.py", content=code, is_test=False),
        ]
        
        task = """Initialize a git repository, commit the initial feature.py file, then update the version string from 'v1' to 'v2' and commit that change. Track your work with git throughout."""
        
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
            metadata={
                "scenario_type": "git",
                "command_focus": "git",
                "solution_steps": [
                    "Initialize repo: git init",
                    "Stage file: git add feature.py",
                    "Initial commit: git commit -m 'Initial commit'",
                    "Make change: sed -i 's/v1/v2/g' feature.py",
                    "View changes: git diff",
                    "Stage changes: git add feature.py",
                    "Commit update: git commit -m 'Update to v2'",
                    "View history: git log --oneline"
                ]
            }
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
        
        task = """The text processor has a bug - it's treating words with different cases as different words (e.g., 'apple' and 'Apple'). Fix the code to handle case-insensitive processing."""
        
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
            metadata={
                "scenario_type": "text_transform",
                "command_focus": "tr,sort,uniq",
                "solution_steps": [
                    "View file: cat words.txt",
                    "Convert to lowercase: cat words.txt | tr 'A-Z' 'a-z'",
                    "Sort: cat words.txt | tr 'A-Z' 'a-z' | sort",
                    "Get unique: cat words.txt | tr 'A-Z' 'a-z' | sort | uniq",
                    "Count unique: cat words.txt | tr 'A-Z' 'a-z' | sort | uniq | wc -l",
                    "Identify issue: code doesn't lowercase",
                    "Fix: sed -i 's/f.read().split()/f.read().lower().split()/g' processor.py"
                ]
            }
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
        
        task = """Compare the two fruit files and create a merged.txt file that contains all unique fruits from both files (no duplicates)."""
        
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
            metadata={
                "scenario_type": "comparison",
                "command_focus": "diff,comm",
                "solution_steps": [
                    "Compare files: diff fruits1.txt fruits2.txt",
                    "Unified format: diff -u fruits1.txt fruits2.txt",
                    "Common lines: comm fruits1.txt fruits2.txt",
                    "Merge and deduplicate: cat fruits1.txt fruits2.txt | sort | uniq > merged.txt",
                    "Verify result: cat merged.txt"
                ]
            }
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
        
        task = """Analyze the web server access logs and create a summary.txt file that reports the count of errors (status codes 404 and 500). Explore the log file to understand its format first."""
        
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
            metadata={
                "scenario_type": "log_analysis",
                "command_focus": "awk,cut,grep,pipes",
                "solution_steps": [
                    "View log: cat access.log",
                    "Find errors: grep '404\\|500' access.log",
                    "Extract IPs: cut -d' ' -f1 access.log | sort | uniq",
                    "Count status codes: awk '{print $9}' access.log | sort | uniq -c",
                    "Count GET requests: grep 'GET' access.log | wc -l",
                    "Extract paths: cat access.log | cut -d'\"' -f2 | cut -d' ' -f2",
                    "Create summary: echo \"Error count: $(grep -c '404\\|500' access.log)\" > summary.txt"
                ]
            }
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
        
        task = """Refactor the codebase: rename 'old_function' to 'new_function' everywhere it appears. Make sure to update it in all files where it's used."""
        
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
            metadata={
                "scenario_type": "refactoring",
                "command_focus": "find,xargs,sed",
                "solution_steps": [
                    "Find all occurrences: grep -r 'old_function' .",
                    "Find all Python files: find . -name '*.py'",
                    "Rename in module1: sed -i 's/old_function/new_function/g' module1.py",
                    "Rename in module2: sed -i 's/old_function/new_function/g' module2.py",
                    "Or use find+xargs: find . -name '*.py' -exec sed -i 's/old_function/new_function/g' {} \\;",
                    "Verify: grep -r 'new_function' ."
                ]
            }
        )
    
    def _archive_compression_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario using tar, gzip for archiving."""
        
        file1 = '''def main():
    print("File 1")
'''
        
        file2 = '''def helper():
    return "Helper"
'''
        
        config = '''[settings]
debug=true
port=8000
'''
        
        files = [
            FileContent(path="src/main.py", content=file1, is_test=False),
            FileContent(path="src/helper.py", content=file2, is_test=False),
            FileContent(path="config.ini", content=config, is_test=False),
        ]
        
        task = """Create a compressed backup archive named 'backup.tar.gz' containing all Python files in the 'src' directory and the config.ini file. Then verify the archive contents."""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="text_match", target=".",
                               expected="backup.tar.gz", description="Archive created")
            ],
            expected_commands=6,
            cli_history=["ls", "ls src/"],
            metadata={
                "scenario_type": "archive",
                "command_focus": "tar,gzip,find",
                "solution_steps": [
                    "List files: ls -R",
                    "Create archive: tar -czf backup.tar.gz src/ config.ini",
                    "Or find and archive: find . -name '*.py' -o -name '*.ini' | tar -czf backup.tar.gz -T -",
                    "List archive contents: tar -tzf backup.tar.gz",
                    "Verify: ls -lh backup.tar.gz",
                    "Extract to test (optional): tar -xzf backup.tar.gz -C /tmp/test"
                ]
            }
        )
    
    def _batch_processing_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario using find + xargs for batch operations."""
        
        files_content = [
            ("data/file1.txt", "TODO: Review this\nSome content\nFIXME: Bug here"),
            ("data/file2.txt", "Clean content\nNo issues"),
            ("data/file3.txt", "TODO: Update docs\nMore content"),
            ("other/notes.txt", "TODO: Remember this"),
        ]
        
        files = [FileContent(path=path, content=content, is_test=False) 
                for path, content in files_content]
        
        task = """Find all .txt files in the 'data' directory that contain 'TODO' and create a report.txt file listing the filenames and the count of TODO items in each file."""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="text_match", target="report.txt",
                               expected="file", description="Report created")
            ],
            expected_commands=8,
            cli_history=["ls", "ls data/"],
            metadata={
                "scenario_type": "batch_processing",
                "command_focus": "find,xargs,grep",
                "solution_steps": [
                    "Find txt files in data: find data/ -name '*.txt'",
                    "Search for TODO in each: find data/ -name '*.txt' -exec grep -l 'TODO' {} \\;",
                    "Count TODOs per file: find data/ -name '*.txt' -exec sh -c 'echo \"{}:\" $(grep -c TODO {})' \\;",
                    "Or use xargs: find data/ -name '*.txt' | xargs grep -c TODO",
                    "Create report: find data/ -name '*.txt' -exec grep -c TODO {} + > report.txt",
                    "Better format: find data/ -name '*.txt' -print0 | xargs -0 grep -l TODO | tee report.txt",
                    "Verify: cat report.txt"
                ]
            }
        )
    
    def _complex_redirection_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario using complex I/O redirection."""
        
        script = '''#!/usr/bin/env python3
import sys

print("Standard output message")
print("Error message", file=sys.stderr)
print("Another output")
print("Another error", file=sys.stderr)
'''
        
        files = [
            FileContent(path="script.py", content=script, is_test=False),
        ]
        
        task = """Run the script and separate the output: save standard output to 'output.log', errors to 'errors.log', and create a combined log 'all.log' with both. Verify all three files exist with the correct content."""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="text_match", target="output.log",
                               expected="output", description="Output log created"),
                VerificationRule(type="text_match", target="errors.log",
                               expected="Error", description="Error log created"),
            ],
            expected_commands=8,
            cli_history=["cat script.py"],
            metadata={
                "scenario_type": "redirection",
                "command_focus": "redirection,pipes",
                "solution_steps": [
                    "Run with stdout redirect: python3 script.py > output.log",
                    "Run with stderr redirect: python3 script.py 2> errors.log",
                    "Run with both: python3 script.py > output.log 2> errors.log",
                    "Run with combined: python3 script.py &> all.log",
                    "Or: python3 script.py > all.log 2>&1",
                    "Verify output: cat output.log",
                    "Verify errors: cat errors.log",
                    "Verify combined: cat all.log"
                ]
            }
        )
    
    def _symbolic_links_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario using symbolic links."""
        
        config_dev = '''[database]
host=localhost
port=5432
name=dev_db
'''
        
        config_prod = '''[database]
host=prod.server.com
port=5432
name=prod_db
'''
        
        app = '''import configparser

config = configparser.ConfigParser()
config.read('config.ini')  # Reads the symlink

print(f"Database: {config['database']['name']}")
'''
        
        files = [
            FileContent(path="config.dev.ini", content=config_dev, is_test=False),
            FileContent(path="config.prod.ini", content=config_prod, is_test=False),
            FileContent(path="app.py", content=app, is_test=False),
        ]
        
        task = """Create a symbolic link named 'config.ini' that points to 'config.dev.ini'. Then verify the link works by running the app and checking that it uses the dev configuration."""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="execution", target="app.py",
                               description="App runs successfully")
            ],
            expected_commands=6,
            cli_history=["ls *.ini"],
            metadata={
                "scenario_type": "symlinks",
                "command_focus": "ln,readlink,ls",
                "solution_steps": [
                    "Create symlink: ln -s config.dev.ini config.ini",
                    "Verify link: ls -l config.ini",
                    "Check target: readlink config.ini",
                    "Test app: python3 app.py",
                    "Switch to prod: rm config.ini && ln -s config.prod.ini config.ini",
                    "Verify switch: readlink config.ini"
                ]
            }
        )
    
    def _permissions_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Scenario using chmod for permissions."""
        
        script = '''#!/bin/bash
echo "Running deployment script..."
'''
        
        deploy_py = '''#!/usr/bin/env python3
print("Deploying application...")
'''
        
        readme = '''# Deployment Scripts

Run deploy.sh to start deployment.
'''
        
        files = [
            FileContent(path="deploy.sh", content=script, is_test=False),
            FileContent(path="deploy.py", content=deploy_py, is_test=False),
            FileContent(path="README.md", content=readme, is_test=False),
        ]
        
        task = """Make the deploy.sh and deploy.py scripts executable. The README should remain read-only. Verify the permissions are set correctly."""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="text_match", target=".",
                               expected="deploy", description="Scripts exist")
            ],
            expected_commands=7,
            cli_history=["ls -l"],
            metadata={
                "scenario_type": "permissions",
                "command_focus": "chmod,ls",
                "solution_steps": [
                    "Check current permissions: ls -l",
                    "Make deploy.sh executable: chmod +x deploy.sh",
                    "Make deploy.py executable: chmod +x deploy.py",
                    "Or both at once: chmod +x deploy.*",
                    "Make README read-only: chmod 444 README.md",
                    "Verify: ls -l",
                    "Test execution: ./deploy.sh"
                ]
            }
        )
    
    def _data_pipeline_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Complex multi-step data processing pipeline."""
        
        access_log = '''192.168.1.10 - - [30/Oct/2024:10:15:23] "GET /api/users HTTP/1.1" 200 1234
192.168.1.11 - - [30/Oct/2024:10:15:24] "POST /api/login HTTP/1.1" 200 567
192.168.1.10 - - [30/Oct/2024:10:15:25] "GET /api/profile HTTP/1.1" 200 890
192.168.1.12 - - [30/Oct/2024:10:15:26] "GET /api/users HTTP/1.1" 200 1234
192.168.1.11 - - [30/Oct/2024:10:15:27] "DELETE /api/session HTTP/1.1" 204 0
192.168.1.10 - - [30/Oct/2024:10:15:28] "GET /api/users HTTP/1.1" 200 1234
'''
        
        files = [
            FileContent(path="access.log", content=access_log, is_test=False),
        ]
        
        task = """Process the access log to create 'top_ips.txt' containing the top 3 IP addresses by request count, sorted by frequency. Each line should show the count and IP address."""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="text_match", target="top_ips.txt",
                               expected="192.168", description="Top IPs listed")
            ],
            expected_commands=10,
            cli_history=["head access.log"],
            metadata={
                "scenario_type": "data_pipeline",
                "command_focus": "cut,sort,uniq,head,pipes",
                "solution_steps": [
                    "View log: cat access.log",
                    "Extract IPs: cut -d' ' -f1 access.log",
                    "Sort IPs: cut -d' ' -f1 access.log | sort",
                    "Count occurrences: cut -d' ' -f1 access.log | sort | uniq -c",
                    "Sort by count: cut -d' ' -f1 access.log | sort | uniq -c | sort -rn",
                    "Get top 3: cut -d' ' -f1 access.log | sort | uniq -c | sort -rn | head -3",
                    "Save to file: cut -d' ' -f1 access.log | sort | uniq -c | sort -rn | head -3 > top_ips.txt",
                    "Verify: cat top_ips.txt"
                ]
            }
        )
    
    def _config_editing_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Complex configuration file editing with sed."""
        
        config = '''# Application Configuration
DEBUG=false
LOG_LEVEL=info
DATABASE_HOST=localhost
DATABASE_PORT=5432
# API Configuration
API_KEY=old_key_12345
API_TIMEOUT=30
# Cache Settings
CACHE_ENABLED=false
CACHE_TTL=3600
'''
        
        files = [
            FileContent(path="config.env", content=config, is_test=False),
        ]
        
        task = """Update the configuration file: enable DEBUG mode, change LOG_LEVEL to 'debug', enable CACHE, update API_KEY to 'new_key_67890', and add a comment '# Updated for development' at the top."""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="text_match", target="config.env",
                               expected="DEBUG=true", description="DEBUG enabled"),
                VerificationRule(type="text_match", target="config.env",
                               expected="new_key", description="API key updated"),
            ],
            expected_commands=8,
            cli_history=["cat config.env"],
            metadata={
                "scenario_type": "config_editing",
                "command_focus": "sed,grep",
                "solution_steps": [
                    "Backup: cp config.env config.env.bak",
                    "Enable DEBUG: sed -i 's/DEBUG=false/DEBUG=true/g' config.env",
                    "Change LOG_LEVEL: sed -i 's/LOG_LEVEL=info/LOG_LEVEL=debug/g' config.env",
                    "Enable CACHE: sed -i 's/CACHE_ENABLED=false/CACHE_ENABLED=true/g' config.env",
                    "Update API_KEY: sed -i 's/API_KEY=old_key_12345/API_KEY=new_key_67890/g' config.env",
                    "Add comment at top: sed -i '1i# Updated for development' config.env",
                    "Verify changes: cat config.env",
                    "Or check specific: grep -E 'DEBUG|LOG_LEVEL|CACHE_ENABLED|API_KEY' config.env"
                ]
            }
        )
    
    def _directory_tree_scenario(self, difficulty: DifficultyLevel, language: str) -> Scenario:
        """Complex find operations on directory trees."""
        
        files_content = [
            ("src/main.py", "# Main module\nprint('main')"),
            ("src/utils.py", "# Utils\ndef helper(): pass"),
            ("tests/test_main.py", "# Tests\nimport main"),
            ("tests/test_utils.py", "# Tests\nimport utils"),
            ("docs/README.md", "# Documentation"),
            ("docs/API.md", "# API Docs"),
            (".gitignore", "*.pyc\n__pycache__/"),
            ("setup.py", "from setuptools import setup"),
        ]
        
        files = [FileContent(path=path, content=content, is_test=False) 
                for path, content in files_content]
        
        task = """Find all Python files (*.py) in the project, excluding the 'tests' directory, and create a 'python_files.txt' listing with their full paths. Also create 'file_count.txt' with the total count."""
        
        return Scenario(
            difficulty=difficulty,
            language="python",
            task_description=task,
            files=files,
            verification_rules=[
                VerificationRule(type="text_match", target="python_files.txt",
                               expected="src/", description="Python files listed"),
            ],
            expected_commands=8,
            cli_history=["ls", "tree ."],
            metadata={
                "scenario_type": "directory_tree",
                "command_focus": "find,wc,grep",
                "solution_steps": [
                    "List all directories: find . -type d",
                    "Find all .py files: find . -name '*.py'",
                    "Exclude tests: find . -name '*.py' -not -path './tests/*'",
                    "Save to file: find . -name '*.py' -not -path './tests/*' > python_files.txt",
                    "Count files: find . -name '*.py' -not -path './tests/*' | wc -l",
                    "Save count: find . -name '*.py' -not -path './tests/*' | wc -l > file_count.txt",
                    "Verify list: cat python_files.txt",
                    "Verify count: cat file_count.txt"
                ]
            }
        )

