"""Generate diverse training prompts for LLM training."""

import random
import json
from typing import List, Dict, Any
from pathlib import Path
from collections import Counter

from cli_rl_env.scenario_generator.base import DifficultyLevel
from cli_rl_env.scenario_generator.python_generator import PythonScenarioGenerator
from cli_rl_env.scenario_generator.javascript_generator import JavaScriptScenarioGenerator
from cli_rl_env.scenario_generator.diverse_scenarios import DiverseScenarioGenerator


class PromptDatasetGenerator:
    """Generate large diverse datasets of training prompts."""
    
    def __init__(self, seed: int = None):
        """Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed:
            random.seed(seed)
        
        self.python_gen = PythonScenarioGenerator(seed=seed)
        self.js_gen = JavaScriptScenarioGenerator(seed=seed)
        self.diverse_gen = DiverseScenarioGenerator(seed=seed)
        self.command_coverage = Counter()
    
    def get_all_scenario_types(self) -> Dict[str, List[str]]:
        """Get all available scenario types from all generators.
        
        Returns:
            Dictionary mapping generator name to list of scenario types
        """
        return {
            'python_generator': [
                'calculator',
                'data_processor', 
                'string_utils',
                'algorithms'
            ],
            'javascript_generator': [
                'utils',
                'array_ops',
                'validators'
            ],
            'diverse_scenarios': [
                'grep_intensive',
                'sed_intensive',
                'awk_cut',
                'piping',
                'multi_file_operations',
                'git_workflow',
                'text_transformation',
                'file_comparison',
                'log_analysis',
                'refactoring',
                'archive_compression',
                'batch_processing',
                'complex_redirection',
                'symbolic_links',
                'permissions',
                'data_pipeline',
                'config_editing',
                'directory_tree'
            ]
        }
    
    def generate_dataset(
        self,
        num_prompts: int = 1000,
        difficulty_distribution: Dict[str, float] = None,
        generator_mix: Dict[str, float] = None,
        output_file: str = None
    ) -> List[Dict[str, Any]]:
        """Generate a large dataset of training prompts using ALL scenario generators.
        
        This method now uses all three generators (python, javascript, and diverse)
        to ensure maximum scenario variety and CLI command coverage.
        
        Args:
            num_prompts: Number of prompts to generate
            difficulty_distribution: Distribution of difficulties
                Default: {'easy': 0.1, 'medium': 0.2, 'hard': 0.4, 'very_hard': 0.3}
            generator_mix: Distribution of generator types
                Default: {'python': 0.25, 'javascript': 0.25, 'diverse': 0.5}
                'diverse' scenarios focus on CLI command diversity
            output_file: Optional file to save dataset to
            
        Returns:
            List of prompt dictionaries
        """
        if difficulty_distribution is None:
            # Focus on harder prompts for training
            difficulty_distribution = {
                'easy': 0.1,
                'medium': 0.2,
                'hard': 0.4,
                'very_hard': 0.3
            }
        
        if generator_mix is None:
            # Default: 50% diverse scenarios for command coverage
            generator_mix = {
                'python': 0.25,
                'javascript': 0.25,
                'diverse': 0.5
            }
        
        # Validate generator_mix
        total = sum(generator_mix.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"generator_mix must sum to 1.0, got {total}")
        
        dataset = []
        difficulties = list(difficulty_distribution.keys())
        diff_weights = list(difficulty_distribution.values())
        
        generators = list(generator_mix.keys())
        gen_weights = list(generator_mix.values())
        
        for i in range(num_prompts):
            # Sample difficulty based on distribution
            difficulty = random.choices(difficulties, weights=diff_weights)[0]
            
            # Sample generator type
            gen_type = random.choices(generators, weights=gen_weights)[0]
            
            # Generate scenario based on type
            if gen_type == 'python':
                scenario = self.python_gen.generate(DifficultyLevel(difficulty))
            elif gen_type == 'javascript':
                scenario = self.js_gen.generate(DifficultyLevel(difficulty))
            else:  # diverse
                language = random.choice(['python', 'javascript'])
                scenario = self.diverse_gen.generate_diverse_scenario(
                    DifficultyLevel(difficulty), language
                )
                # Track command usage
                if 'command_focus' in scenario.metadata:
                    for cmd in scenario.metadata['command_focus'].split(','):
                        self.command_coverage[cmd.strip()] += 1
            
            # Create training example
            prompt_data = self._scenario_to_prompt_data(scenario, f'prompt_{i:06d}')
            dataset.append(prompt_data)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_prompts} prompts...")
        
        print(f"\nTotal: {len(dataset)} prompts generated")
        if self.command_coverage:
            print(f"Command coverage: {len(self.command_coverage)} unique commands tracked")
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            print(f"Dataset saved to {output_file}")
        
        return dataset
    
    def generate_balanced_diverse_dataset(
        self,
        num_prompts: int = 1000,
        difficulty_distribution: Dict[str, float] = None,
        diverse_scenario_ratio: float = 0.6,
        output_file: str = None
    ) -> List[Dict[str, Any]]:
        """Generate a balanced dataset with high command diversity.
        
        This method ensures that all CLI commands are well-represented in the dataset
        by using specialized diverse scenario generators for a significant portion of
        the dataset.
        
        Args:
            num_prompts: Total number of prompts to generate
            difficulty_distribution: Distribution of difficulties (must sum to 1.0)
            diverse_scenario_ratio: Proportion of diverse scenarios (0.0-1.0, default 0.6 = 60%)
            output_file: Optional file to save dataset to
            
        Returns:
            List of prompt dictionaries with high command diversity
            
        Raises:
            ValueError: If diverse_scenario_ratio not in [0.0, 1.0] or difficulty_distribution invalid
        """
        # Validate diverse_scenario_ratio
        if not 0.0 <= diverse_scenario_ratio <= 1.0:
            raise ValueError(
                f"diverse_scenario_ratio must be between 0.0 and 1.0, got {diverse_scenario_ratio}"
            )
        
        if difficulty_distribution is None:
            difficulty_distribution = {
                'easy': 0.1,
                'medium': 0.2,
                'hard': 0.4,
                'very_hard': 0.3
            }
        else:
            # Validate difficulty distribution
            total = sum(difficulty_distribution.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Difficulty distribution must sum to 1.0, got {total:.3f}. "
                    f"Provided: {difficulty_distribution}"
                )
            
            # Validate keys
            valid_difficulties = {'easy', 'medium', 'hard', 'very_hard'}
            invalid = set(difficulty_distribution.keys()) - valid_difficulties
            if invalid:
                raise ValueError(
                    f"Invalid difficulties: {invalid}. "
                    f"Must be one of: {valid_difficulties}"
                )
        
        dataset = []
        difficulties = list(difficulty_distribution.keys())
        weights = list(difficulty_distribution.values())
        
        # Calculate split
        num_diverse = int(num_prompts * diverse_scenario_ratio)
        num_standard = num_prompts - num_diverse
        
        print(f"Generating {num_diverse} diverse scenarios and {num_standard} standard scenarios...")
        
        # Generate diverse scenarios (60% by default)
        for i in range(num_diverse):
            difficulty = random.choices(difficulties, weights=weights)[0]
            language = random.choice(['python', 'javascript'])
            
            scenario = self.diverse_gen.generate_diverse_scenario(
                DifficultyLevel(difficulty), language
            )
            
            prompt_data = self._scenario_to_prompt_data(scenario, f'diverse_{i:06d}')
            dataset.append(prompt_data)
            
            # Track command usage
            if 'command_focus' in scenario.metadata:
                for cmd in scenario.metadata['command_focus'].split(','):
                    self.command_coverage[cmd.strip()] += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Diverse: {i + 1}/{num_diverse} generated...")
        
        # Generate standard scenarios (40% by default)
        for i in range(num_standard):
            difficulty = random.choices(difficulties, weights=weights)[0]
            language = random.choice(['python', 'javascript'])
            
            if language == 'python':
                scenario = self.python_gen.generate(DifficultyLevel(difficulty))
            else:
                scenario = self.js_gen.generate(DifficultyLevel(difficulty))
            
            prompt_data = self._scenario_to_prompt_data(scenario, f'standard_{i:06d}')
            dataset.append(prompt_data)
            
            if (i + 1) % 100 == 0:
                print(f"  Standard: {i + 1}/{num_standard} generated...")
        
        # Shuffle the combined dataset
        random.shuffle(dataset)
        
        # Re-assign sequential IDs
        for i, item in enumerate(dataset):
            item['id'] = f'prompt_{i:06d}'
        
        print(f"\nTotal: {len(dataset)} prompts generated")
        print(f"Command coverage: {len(self.command_coverage)} unique commands")
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            print(f"Dataset saved to {output_file}")
        
        return dataset
    
    def _scenario_to_prompt_data(self, scenario, prompt_id: str) -> Dict[str, Any]:
        """Convert a Scenario object to a prompt data dictionary."""
        return {
            'id': prompt_id,
            'difficulty': scenario.difficulty.value,
            'language': scenario.language,
            'task_description': scenario.task_description,
            'files': [
                {
                    'path': f.path,
                    'content': f.content,
                    'is_test': f.is_test
                }
                for f in scenario.files
            ],
            'cli_history': scenario.cli_history,
            'expected_commands': scenario.expected_commands,
            'verification_rules': [
                {
                    'type': v.type,
                    'target': v.target,
                    'description': v.description
                }
                for v in scenario.verification_rules
            ],
            'metadata': scenario.metadata
        }
    
    def generate_advanced_scenarios(self, num_prompts: int = 500) -> List[Dict[str, Any]]:
        """Generate advanced scenarios with complex requirements.
        
        These are harder scenarios that require:
        - Multi-step reasoning
        - Understanding code architecture
        - Complex text transformations
        - Error handling
        
        Args:
            num_prompts: Number of advanced prompts
            
        Returns:
            List of advanced prompt dictionaries
        """
        dataset = []
        
        scenario_templates = [
            self._multi_file_refactor,
            self._complex_bug_hunt,
            self._architectural_change,
            self._performance_optimization,
            self._security_fix,
            self._test_driven_development,
            self._dependency_update,
            self._code_review_fixes,
        ]
        
        for i in range(num_prompts):
            template = random.choice(scenario_templates)
            language = random.choice(['python', 'javascript'])
            
            prompt_data = template(language, i)
            dataset.append(prompt_data)
            
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{num_prompts} advanced prompts...")
        
        return dataset
    
    def _multi_file_refactor(self, language: str, idx: int) -> Dict[str, Any]:
        """Generate multi-file refactoring scenario."""
        if language == 'python':
            return {
                'id': f'adv_refactor_py_{idx:06d}',
                'difficulty': 'very_hard',
                'language': 'python',
                'task_description': (
                    "Refactor the codebase to extract the validation logic into a separate "
                    "validators.py module. Update all imports and ensure tests still pass. "
                    "The validation functions should be moved from utils.py to the new module."
                ),
                'scenario_type': 'multi_file_refactor',
                'expected_commands': 15
            }
        else:
            return {
                'id': f'adv_refactor_js_{idx:06d}',
                'difficulty': 'very_hard',
                'language': 'javascript',
                'task_description': (
                    "Split the monolithic utils.js file into separate modules: validators.js, "
                    "formatters.js, and helpers.js. Update all require statements in the tests."
                ),
                'scenario_type': 'multi_file_refactor',
                'expected_commands': 15
            }
    
    def _complex_bug_hunt(self, language: str, idx: int) -> Dict[str, Any]:
        """Generate complex bug hunting scenario."""
        bug_types = [
            "race condition", "off-by-one error", "incorrect error handling",
            "edge case failure", "type coercion issue"
        ]
        bug = random.choice(bug_types)
        
        return {
            'id': f'adv_bug_{language}_{idx:06d}',
            'difficulty': 'hard',
            'language': language,
            'task_description': (
                f"The code has a subtle {bug} that only manifests in specific conditions. "
                f"Run the tests to identify the failure, trace through the code to find "
                f"the root cause, and fix it. Multiple functions may need to be examined."
            ),
            'scenario_type': 'complex_bug_hunt',
            'expected_commands': 12
        }
    
    def _architectural_change(self, language: str, idx: int) -> Dict[str, Any]:
        """Generate architectural change scenario."""
        if language == 'python':
            return {
                'id': f'adv_arch_py_{idx:06d}',
                'difficulty': 'very_hard',
                'language': 'python',
                'task_description': (
                    "Refactor the code to use dependency injection instead of global state. "
                    "The current implementation uses module-level variables which makes "
                    "testing difficult. Create a proper class-based design with dependency "
                    "injection while maintaining backward compatibility."
                ),
                'scenario_type': 'architectural_change',
                'expected_commands': 20
            }
        else:
            return {
                'id': f'adv_arch_js_{idx:06d}',
                'difficulty': 'very_hard',
                'language': 'javascript',
                'task_description': (
                    "Convert the callback-based async code to use Promises and async/await. "
                    "Update all function signatures and ensure error handling is preserved."
                ),
                'scenario_type': 'architectural_change',
                'expected_commands': 18
            }
    
    def _performance_optimization(self, language: str, idx: int) -> Dict[str, Any]:
        """Generate performance optimization scenario."""
        issues = [
            "O(n²) algorithm that should be O(n)",
            "repeated database queries in a loop",
            "unnecessary object creation",
            "missing memoization"
        ]
        issue = random.choice(issues)
        
        return {
            'id': f'adv_perf_{language}_{idx:06d}',
            'difficulty': 'hard',
            'language': language,
            'task_description': (
                f"The code has a performance issue: {issue}. "
                f"Identify the bottleneck and optimize it while maintaining correctness. "
                f"All existing tests must still pass."
            ),
            'scenario_type': 'performance_optimization',
            'expected_commands': 10
        }
    
    def _security_fix(self, language: str, idx: int) -> Dict[str, Any]:
        """Generate security fix scenario."""
        vulnerabilities = [
            "SQL injection vulnerability",
            "path traversal vulnerability",
            "XSS vulnerability in user input",
            "insecure random number generation"
        ]
        vuln = random.choice(vulnerabilities)
        
        return {
            'id': f'adv_sec_{language}_{idx:06d}',
            'difficulty': 'very_hard',
            'language': language,
            'task_description': (
                f"SECURITY ALERT: The code contains a {vuln}. "
                f"Identify the vulnerable code path, implement proper sanitization/validation, "
                f"and add tests to verify the fix prevents exploitation."
            ),
            'scenario_type': 'security_fix',
            'expected_commands': 12
        }
    
    def _test_driven_development(self, language: str, idx: int) -> Dict[str, Any]:
        """Generate TDD scenario."""
        return {
            'id': f'adv_tdd_{language}_{idx:06d}',
            'difficulty': 'hard',
            'language': language,
            'task_description': (
                "The tests are written but the implementation is missing or incomplete. "
                "Read the test file to understand the requirements, then implement the "
                "functions to make all tests pass. Follow the test specifications exactly."
            ),
            'scenario_type': 'test_driven_development',
            'expected_commands': 8
        }
    
    def _dependency_update(self, language: str, idx: int) -> Dict[str, Any]:
        """Generate dependency update scenario."""
        return {
            'id': f'adv_dep_{language}_{idx:06d}',
            'difficulty': 'very_hard',
            'language': language,
            'task_description': (
                "A dependency has been updated with breaking API changes. The code needs "
                "to be updated to work with the new API while maintaining all functionality. "
                "Check the import statements and function calls that use the old API."
            ),
            'scenario_type': 'dependency_update',
            'expected_commands': 15
        }
    
    def _code_review_fixes(self, language: str, idx: int) -> Dict[str, Any]:
        """Generate code review fix scenario."""
        issues = [
            "inconsistent naming conventions",
            "missing error handling",
            "inadequate input validation",
            "poor code documentation"
        ]
        issue = random.choice(issues)
        
        return {
            'id': f'adv_review_{language}_{idx:06d}',
            'difficulty': 'medium',
            'language': language,
            'task_description': (
                f"Code review found the following issue: {issue}. "
                f"Fix all instances throughout the codebase while maintaining functionality. "
                f"Ensure tests still pass after your changes."
            ),
            'scenario_type': 'code_review_fixes',
            'expected_commands': 10
        }
    
    def save_dataset_splits(
        self,
        dataset: List[Dict[str, Any]],
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ):
        """Split dataset into train/val/test and save.
        
        Args:
            dataset: Full dataset
            output_dir: Directory to save splits
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
        
        # Shuffle dataset
        dataset = dataset.copy()
        random.shuffle(dataset)
        
        n = len(dataset)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_set = dataset[:train_end]
        val_set = dataset[train_end:val_end]
        test_set = dataset[val_end:]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        with open(output_path / 'train.json', 'w') as f:
            json.dump(train_set, f, indent=2)
        
        with open(output_path / 'val.json', 'w') as f:
            json.dump(val_set, f, indent=2)
        
        with open(output_path / 'test.json', 'w') as f:
            json.dump(test_set, f, indent=2)
        
        # Save statistics
        stats = {
            'total': n,
            'train': len(train_set),
            'val': len(val_set),
            'test': len(test_set),
            'difficulty_distribution': self._get_difficulty_dist(dataset),
            'language_distribution': self._get_language_dist(dataset)
        }
        
        with open(output_path / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nDataset splits saved to {output_dir}/")
        print(f"  Train: {len(train_set)} examples")
        print(f"  Val: {len(val_set)} examples")
        print(f"  Test: {len(test_set)} examples")
    
    def _get_difficulty_dist(self, dataset: List[Dict]) -> Dict[str, int]:
        """Get difficulty distribution."""
        dist = {}
        for item in dataset:
            diff = item['difficulty']
            dist[diff] = dist.get(diff, 0) + 1
        return dist
    
    def _get_language_dist(self, dataset: List[Dict]) -> Dict[str, int]:
        """Get language distribution."""
        dist = {}
        for item in dataset:
            lang = item['language']
            dist[lang] = dist.get(lang, 0) + 1
        return dist

