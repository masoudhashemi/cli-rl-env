"""Generate JavaScript code scenarios with bugs."""

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


class JavaScriptScenarioGenerator(ScenarioGenerator):
    """Generate JavaScript debugging scenarios."""
    
    def get_language(self) -> str:
        """Get the programming language."""
        return "javascript"
    
    def generate(self, difficulty: DifficultyLevel) -> Scenario:
        """Generate a JavaScript scenario.
        
        Args:
            difficulty: Target difficulty level
            
        Returns:
            Complete scenario
        """
        scenario_type = random.choice(['utils', 'array_ops', 'validators'])
        
        if scenario_type == 'utils':
            return self._generate_utils_scenario(difficulty)
        elif scenario_type == 'array_ops':
            return self._generate_array_ops_scenario(difficulty)
        else:
            return self._generate_validators_scenario(difficulty)
    
    def _generate_utils_scenario(self, difficulty: DifficultyLevel) -> Scenario:
        """Generate a utilities scenario."""
        main_code = '''// Utility functions

function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}

function isEven(n) {
    return n % 2 === 0;
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function range(start, end) {
    const result = [];
    for (let i = start; i <= end; i++) {
        result.push(i);
    }
    return result;
}

module.exports = { add, multiply, isEven, capitalize, range };
'''
        
        test_code = '''// Tests for utility functions

const { add, multiply, isEven, capitalize, range } = require('./utils');

function assertEquals(actual, expected, message) {
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
        throw new Error(`${message}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
    }
}

function test_add() {
    assertEquals(add(2, 3), 5, "add(2, 3)");
    assertEquals(add(-1, 1), 0, "add(-1, 1)");
    console.log("✓ test_add passed");
}

function test_multiply() {
    assertEquals(multiply(4, 5), 20, "multiply(4, 5)");
    assertEquals(multiply(-2, 3), -6, "multiply(-2, 3)");
    console.log("✓ test_multiply passed");
}

function test_isEven() {
    assertEquals(isEven(4), true, "isEven(4)");
    assertEquals(isEven(5), false, "isEven(5)");
    console.log("✓ test_isEven passed");
}

function test_capitalize() {
    assertEquals(capitalize("hello"), "Hello", "capitalize('hello')");
    assertEquals(capitalize("world"), "World", "capitalize('world')");
    console.log("✓ test_capitalize passed");
}

function test_range() {
    assertEquals(range(1, 5), [1, 2, 3, 4, 5], "range(1, 5)");
    assertEquals(range(0, 0), [0], "range(0, 0)");
    console.log("✓ test_range passed");
}

// Run all tests
try {
    test_add();
    test_multiply();
    test_isEven();
    test_capitalize();
    test_range();
    console.log("All tests passed!");
} catch (e) {
    console.error("Test failed:", e.message);
    process.exit(1);
}
'''
        
        num_bugs = {
            DifficultyLevel.EASY: 1,
            DifficultyLevel.MEDIUM: 2,
            DifficultyLevel.HARD: 3,
            DifficultyLevel.VERY_HARD: 4
        }[difficulty]
        
        buggy_code, bug_descriptions = BugInjector.inject_javascript_bugs(
            main_code, num_bugs
        )
        
        files = [
            FileContent(path="utils.js", content=buggy_code, is_test=False),
            FileContent(path="test_utils.js", content=test_code, is_test=True),
        ]
        
        cli_history = self._generate_cli_history(difficulty, files)
        
        task_description = PromptGenerator.generate_debug_prompt(
            language="javascript",
            bug_descriptions=bug_descriptions,
            difficulty=difficulty,
            file_structure=[f.path for f in files]
        )
        
        verification_rules = [
            VerificationRule(
                type="execution",
                target="test_utils.js",
                expected=0,
                description="Tests must run successfully (exit code 0)"
            )
        ]
        
        expected_commands = num_bugs * 3
        
        return Scenario(
            difficulty=difficulty,
            language="javascript",
            task_description=task_description,
            files=files,
            verification_rules=verification_rules,
            expected_commands=expected_commands,
            cli_history=cli_history,
            metadata={"bugs": bug_descriptions, "scenario_type": "utils"}
        )
    
    def _generate_array_ops_scenario(self, difficulty: DifficultyLevel) -> Scenario:
        """Generate an array operations scenario."""
        main_code = '''// Array operation functions

function sum(arr) {
    return arr.reduce((a, b) => a + b, 0);
}

function findMax(arr) {
    if (arr.length === 0) return null;
    return Math.max(...arr);
}

function removeDuplicates(arr) {
    return [...new Set(arr)];
}

function flatten(arr) {
    return arr.flat();
}

function chunk(arr, size) {
    const result = [];
    for (let i = 0; i < arr.length; i += size) {
        result.push(arr.slice(i, i + size));
    }
    return result;
}

module.exports = { sum, findMax, removeDuplicates, flatten, chunk };
'''
        
        test_code = '''// Tests for array operations

const { sum, findMax, removeDuplicates, flatten, chunk } = require('./array_ops');

function assertEquals(actual, expected, message) {
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
        throw new Error(`${message}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
    }
}

function test_sum() {
    assertEquals(sum([1, 2, 3, 4, 5]), 15, "sum([1, 2, 3, 4, 5])");
    assertEquals(sum([]), 0, "sum([])");
    console.log("✓ test_sum passed");
}

function test_findMax() {
    assertEquals(findMax([1, 5, 3, 9, 2]), 9, "findMax([1, 5, 3, 9, 2])");
    assertEquals(findMax([]), null, "findMax([])");
    console.log("✓ test_findMax passed");
}

function test_removeDuplicates() {
    assertEquals(removeDuplicates([1, 2, 2, 3, 1, 4]), [1, 2, 3, 4], "removeDuplicates");
    console.log("✓ test_removeDuplicates passed");
}

function test_flatten() {
    assertEquals(flatten([[1, 2], [3, 4], [5]]), [1, 2, 3, 4, 5], "flatten");
    console.log("✓ test_flatten passed");
}

function test_chunk() {
    assertEquals(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]], "chunk");
    console.log("✓ test_chunk passed");
}

// Run all tests
try {
    test_sum();
    test_findMax();
    test_removeDuplicates();
    test_flatten();
    test_chunk();
    console.log("All tests passed!");
} catch (e) {
    console.error("Test failed:", e.message);
    process.exit(1);
}
'''
        
        num_bugs = {
            DifficultyLevel.EASY: 1,
            DifficultyLevel.MEDIUM: 2,
            DifficultyLevel.HARD: 3,
            DifficultyLevel.VERY_HARD: 4
        }[difficulty]
        
        buggy_code, bug_descriptions = BugInjector.inject_javascript_bugs(
            main_code, num_bugs
        )
        
        files = [
            FileContent(path="array_ops.js", content=buggy_code, is_test=False),
            FileContent(path="test_array_ops.js", content=test_code, is_test=True),
        ]
        
        cli_history = self._generate_cli_history(difficulty, files)
        
        task_description = PromptGenerator.generate_debug_prompt(
            language="javascript",
            bug_descriptions=bug_descriptions,
            difficulty=difficulty,
            file_structure=[f.path for f in files]
        )
        
        verification_rules = [
            VerificationRule(
                type="execution",
                target="test_array_ops.js",
                expected=0,
                description="Tests must run successfully"
            )
        ]
        
        expected_commands = num_bugs * 3
        
        return Scenario(
            difficulty=difficulty,
            language="javascript",
            task_description=task_description,
            files=files,
            verification_rules=verification_rules,
            expected_commands=expected_commands,
            cli_history=cli_history,
            metadata={"bugs": bug_descriptions, "scenario_type": "array_ops"}
        )
    
    def _generate_validators_scenario(self, difficulty: DifficultyLevel) -> Scenario:
        """Generate a validators scenario."""
        main_code = '''// Validation functions

function isValidEmail(email) {
    const regex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return regex.test(email);
}

function isValidPhone(phone) {
    const regex = /^\\d{3}-\\d{3}-\\d{4}$/;
    return regex.test(phone);
}

function isValidPassword(password) {
    return password.length >= 8;
}

function isValidUsername(username) {
    const regex = /^[a-zA-Z0-9_]{3,20}$/;
    return regex.test(username);
}

module.exports = { isValidEmail, isValidPhone, isValidPassword, isValidUsername };
'''
        
        test_code = '''// Tests for validators

const { isValidEmail, isValidPhone, isValidPassword, isValidUsername } = require('./validators');

function assertEquals(actual, expected, message) {
    if (actual !== expected) {
        throw new Error(`${message}: expected ${expected}, got ${actual}`);
    }
}

function test_isValidEmail() {
    assertEquals(isValidEmail("user@example.com"), true, "valid email");
    assertEquals(isValidEmail("invalid"), false, "invalid email");
    console.log("✓ test_isValidEmail passed");
}

function test_isValidPhone() {
    assertEquals(isValidPhone("123-456-7890"), true, "valid phone");
    assertEquals(isValidPhone("1234567890"), false, "invalid phone");
    console.log("✓ test_isValidPhone passed");
}

function test_isValidPassword() {
    assertEquals(isValidPassword("password123"), true, "valid password");
    assertEquals(isValidPassword("short"), false, "invalid password");
    console.log("✓ test_isValidPassword passed");
}

function test_isValidUsername() {
    assertEquals(isValidUsername("user_123"), true, "valid username");
    assertEquals(isValidUsername("ab"), false, "too short username");
    console.log("✓ test_isValidUsername passed");
}

// Run all tests
try {
    test_isValidEmail();
    test_isValidPhone();
    test_isValidPassword();
    test_isValidUsername();
    console.log("All tests passed!");
} catch (e) {
    console.error("Test failed:", e.message);
    process.exit(1);
}
'''
        
        num_bugs = {
            DifficultyLevel.EASY: 1,
            DifficultyLevel.MEDIUM: 2,
            DifficultyLevel.HARD: 3,
            DifficultyLevel.VERY_HARD: 4
        }[difficulty]
        
        buggy_code, bug_descriptions = BugInjector.inject_javascript_bugs(
            main_code, num_bugs
        )
        
        files = [
            FileContent(path="validators.js", content=buggy_code, is_test=False),
            FileContent(path="test_validators.js", content=test_code, is_test=True),
        ]
        
        cli_history = self._generate_cli_history(difficulty, files)
        
        task_description = PromptGenerator.generate_debug_prompt(
            language="javascript",
            bug_descriptions=bug_descriptions,
            difficulty=difficulty,
            file_structure=[f.path for f in files]
        )
        
        verification_rules = [
            VerificationRule(
                type="execution",
                target="test_validators.js",
                expected=0,
                description="Tests must run successfully"
            )
        ]
        
        expected_commands = num_bugs * 3
        
        return Scenario(
            difficulty=difficulty,
            language="javascript",
            task_description=task_description,
            files=files,
            verification_rules=verification_rules,
            expected_commands=expected_commands,
            cli_history=cli_history,
            metadata={"bugs": bug_descriptions, "scenario_type": "validators"}
        )
    
    def _generate_cli_history(self, difficulty: DifficultyLevel, files: List[FileContent]) -> List[str]:
        """Generate simulated CLI history based on difficulty."""
        history = []
        
        if difficulty == DifficultyLevel.EASY:
            history.append("$ ls")
            history.append(" ".join([f.path for f in files]))
        else:
            history.append("$ ls -la")
            for f in files:
                history.append(f"-rw-r--r-- 1 user user {len(f.content)} Oct 30 10:00 {f.path}")
            
            if difficulty in [DifficultyLevel.HARD, DifficultyLevel.VERY_HARD]:
                test_file = [f for f in files if f.is_test][0]
                history.append(f"$ node {test_file.path}")
                history.append("Test failed: ...")
        
        return history

