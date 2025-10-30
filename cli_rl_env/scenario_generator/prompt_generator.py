"""Generate task prompts for scenarios."""

import random
from typing import List
from cli_rl_env.scenario_generator.base import DifficultyLevel


class PromptGenerator:
    """Generates natural language task descriptions."""
    
    @staticmethod
    def generate_debug_prompt(
        language: str,
        bug_descriptions: List[str],
        difficulty: DifficultyLevel,
        file_structure: List[str]
    ) -> str:
        """Generate a debugging task prompt.
        
        Args:
            language: Programming language
            bug_descriptions: List of bugs in the code
            difficulty: Task difficulty
            file_structure: List of file paths
            
        Returns:
            Natural language task description
        """
        intros = [
            f"There's a bug in the {language} code that needs fixing.",
            f"The {language} project has failing tests.",
            f"The code has issues that prevent it from working correctly.",
            f"Fix the broken {language} code.",
        ]
        
        if difficulty == DifficultyLevel.EASY:
            return (
                f"{random.choice(intros)} "
                f"The issue is straightforward - locate the problem and fix it. "
                f"Files: {', '.join(file_structure)}"
            )
        elif difficulty == DifficultyLevel.MEDIUM:
            return (
                f"{random.choice(intros)} "
                f"You'll need to explore the codebase to find the issue. "
                f"Check the test failures for clues. "
                f"Project structure: {', '.join(file_structure)}"
            )
        elif difficulty == DifficultyLevel.HARD:
            return (
                f"{random.choice(intros)} "
                f"Multiple issues may need to be resolved. "
                f"Carefully examine the test output and trace through the code. "
                f"The project has these files: {', '.join(file_structure)}"
            )
        else:  # VERY_HARD
            return (
                f"{random.choice(intros)} "
                f"This is a complex debugging task with multiple related issues. "
                f"You'll need to understand the architecture and trace dependencies. "
                f"Start by running tests to see what's failing. "
                f"Files in project: {', '.join(file_structure)}"
            )
    
    @staticmethod
    def generate_refactor_prompt(
        language: str,
        target_function: str,
        difficulty: DifficultyLevel
    ) -> str:
        """Generate a refactoring task prompt.
        
        Args:
            language: Programming language
            target_function: Name of function to refactor
            difficulty: Task difficulty
            
        Returns:
            Natural language task description
        """
        tasks = [
            f"Refactor the `{target_function}` function to improve readability.",
            f"Extract repeated code in `{target_function}` into helper functions.",
            f"Optimize `{target_function}` while maintaining correctness.",
        ]
        
        return (
            f"{random.choice(tasks)} "
            f"Make sure all tests still pass after your changes."
        )
    
    @staticmethod
    def generate_feature_prompt(
        language: str,
        feature_description: str,
        difficulty: DifficultyLevel
    ) -> str:
        """Generate a feature implementation prompt.
        
        Args:
            language: Programming language
            feature_description: What feature to add
            difficulty: Task difficulty
            
        Returns:
            Natural language task description
        """
        return (
            f"Add a new feature: {feature_description} "
            f"Make sure your implementation passes all tests."
        )

