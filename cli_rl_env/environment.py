"""Main Gymnasium environment for code editing tasks."""

import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cli_rl_env.scenario_generator.base import DifficultyLevel, Scenario
from cli_rl_env.scenario_generator.python_generator import PythonScenarioGenerator
from cli_rl_env.scenario_generator.javascript_generator import JavaScriptScenarioGenerator
from cli_rl_env.executor.sandbox import Sandbox
from cli_rl_env.executor.command_parser import CommandParser
from cli_rl_env.verifier.test_runner import TestRunner
from cli_rl_env.verifier.text_matcher import TextMatcher
from cli_rl_env.verifier.linter import Linter
from cli_rl_env.reward.calculator import RewardCalculator


class CodeEditingEnv(gym.Env):
    """Gymnasium environment for training LLMs on code editing tasks.
    
    This is a single-step environment where the agent:
    1. Receives an observation with task description and file state
    2. Outputs a list of commands with a time estimate
    3. Commands are executed in the sandbox
    4. Verification is performed
    5. Reward is calculated based on correctness and time accuracy
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        difficulty: str = "medium",
        language: Optional[str] = None,
        seed: Optional[int] = None,
        max_commands: int = 50,
        render_mode: Optional[str] = None
    ):
        """Initialize the environment.
        
        Args:
            difficulty: Difficulty level ('easy', 'medium', 'hard', 'very_hard')
            language: Programming language ('python', 'javascript', or None for random)
            seed: Random seed for reproducibility
            max_commands: Maximum number of commands allowed per episode
            render_mode: Render mode for visualization
        """
        super().__init__()
        
        self.difficulty = DifficultyLevel(difficulty)
        self.language = language
        self.seed_value = seed
        self.max_commands = max_commands
        self.render_mode = render_mode
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize generators
        self.python_generator = PythonScenarioGenerator(seed=seed)
        self.js_generator = JavaScriptScenarioGenerator(seed=seed)
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator()
        
        # Episode state
        self.current_scenario: Optional[Scenario] = None
        self.sandbox: Optional[Sandbox] = None
        self.episode_log = []
        
        # Define observation and action spaces
        # Observation is a dict with text fields
        self.observation_space = spaces.Dict({
            'task_description': spaces.Text(max_length=10000),
            'file_tree': spaces.Text(max_length=5000),
            'cli_history': spaces.Text(max_length=10000),
        })
        
        # Action is handled as a dict (commands + time_estimate)
        # In practice, this will be parsed from JSON
        self.action_space = spaces.Dict({
            'commands': spaces.Text(max_length=50000),
            'time_estimate': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32)
        })
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment and generate a new scenario.
        
        Args:
            seed: Random seed
            options: Additional options (can include 'scenario' key for pre-existing scenario)
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Check if a pre-existing scenario is provided in options
        if options and 'scenario' in options:
            self.current_scenario = options['scenario']
        else:
            # Generate new scenario
            language = self.language
            if language is None:
                language = random.choice(['python', 'javascript'])
            
            if language == 'python':
                self.current_scenario = self.python_generator.generate(self.difficulty)
            else:
                self.current_scenario = self.js_generator.generate(self.difficulty)
        
        # Reset episode log
        self.episode_log = []
        
        # Create observation
        observation = self._create_observation()
        
        info = {
            'difficulty': self.difficulty.value,
            'language': self.current_scenario.language,
            'expected_commands': self.current_scenario.expected_commands,
            'scenario_type': self.current_scenario.metadata.get('scenario_type', 'unknown')
        }
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Dict with 'commands' list and 'time_estimate' float
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.current_scenario is None:
            raise RuntimeError("Must call reset() before step()")
        
        # Parse and validate action
        try:
            parsed_action = CommandParser.parse_action(action)
            commands = parsed_action['commands']
            time_estimate = parsed_action['time_estimate']
        except Exception as e:
            # Invalid action format - return negative reward
            return (
                self._create_observation(),
                -100.0,
                True,
                False,
                {'error': f'Invalid action format: {str(e)}'}
            )
        
        # Check command count
        if len(commands) > self.max_commands:
            return (
                self._create_observation(),
                -100.0,
                True,
                False,
                {'error': f'Too many commands: {len(commands)} > {self.max_commands}'}
            )
        
        # Execute commands in sandbox
        with Sandbox(self.current_scenario.files) as sandbox:
            self.sandbox = sandbox
            execution_results = sandbox.execute_commands(commands)
            
            if not execution_results['all_successful']:
                # Commands failed to execute
                reward = -50.0
                info = {
                    'execution_results': execution_results,
                    'error': 'Command execution failed'
                }
                return self._create_observation(), reward, True, False, info
            
            # Run verification
            verification_results = self._run_verification(sandbox)
            
            # Calculate reward
            reward_info = self.reward_calculator.calculate_reward(
                verification_results=verification_results,
                actual_time=execution_results['total_time'],
                estimated_time=time_estimate
            )
            
            reward = reward_info['total_reward']
            
            # Build info dict
            info = {
                'execution_results': execution_results,
                'verification_results': verification_results,
                'reward_breakdown': reward_info,
                'commands_executed': len(commands),
                'expected_commands': self.current_scenario.expected_commands,
                'actual_time': execution_results['total_time'],
                'estimated_time': time_estimate,
            }
            
            # Log results
            self.episode_log.append({
                'action': action,
                'reward': reward,
                'info': info
            })
        
        # Single-step environment - always terminates
        terminated = True
        truncated = False
        
        return self._create_observation(), reward, terminated, truncated, info
    
    def _create_observation(self) -> Dict[str, Any]:
        """Create observation from current scenario.
        
        Returns:
            Observation dict
        """
        if self.current_scenario is None:
            return {
                'task_description': '',
                'file_tree': '',
                'cli_history': ''
            }
        
        # Create file tree representation
        file_tree = "Files:\n"
        for file in self.current_scenario.files:
            file_tree += f"  - {file.path} ({len(file.content)} bytes)\n"
        
        # Format CLI history
        cli_history = "\n".join(self.current_scenario.cli_history)
        
        # Prepare files list for external use (e.g., evaluation)
        files_list = [{'path': f.path, 'content': f.content} for f in self.current_scenario.files]
        
        return {
            'task_description': self.current_scenario.task_description,
            'file_tree': file_tree,
            'cli_history': cli_history,
            'files': files_list  # Include for evaluation module
        }
    
    def _run_verification(self, sandbox: Sandbox) -> Dict[str, Any]:
        """Run all verification rules.
        
        Args:
            sandbox: Sandbox with executed commands
            
        Returns:
            Dict with all verification results
        """
        results = {}
        
        # Run tests
        test_files = [f for f in self.current_scenario.files if f.is_test]
        if test_files:
            test_runner = TestRunner(
                sandbox_path=sandbox.get_sandbox_path(),
                language=self.current_scenario.language
            )
            test_results = test_runner.run_tests(test_files[0].path)
            results['test_results'] = test_results
        
        # Run linting
        non_test_files = [f for f in self.current_scenario.files if not f.is_test]
        if non_test_files:
            linter = Linter(
                sandbox_path=sandbox.get_sandbox_path(),
                language=self.current_scenario.language
            )
            lint_results = linter.lint_file(non_test_files[0].path, strict=False)
            results['lint_results'] = lint_results
        
        # Run text matching rules
        text_matcher = TextMatcher(sandbox_path=sandbox.get_sandbox_path())
        text_match_results = []
        for rule in self.current_scenario.verification_rules:
            if rule.type == 'text_match':
                result = text_matcher.verify_pattern(
                    filepath=rule.target,
                    pattern=rule.expected,
                    should_exist=True
                )
                text_match_results.append(result)
        
        if text_match_results:
            results['text_match_results'] = text_match_results
        
        return results
    
    def render(self):
        """Render the environment state.
        
        Returns:
            Rendered output (depends on render_mode)
        """
        if self.render_mode == "ansi" or self.render_mode == "human":
            output = []
            
            if self.current_scenario:
                output.append("=" * 80)
                output.append(f"Difficulty: {self.current_scenario.difficulty.value}")
                output.append(f"Language: {self.current_scenario.language}")
                output.append("=" * 80)
                output.append("\nTask:")
                output.append(self.current_scenario.task_description)
                output.append("\nFiles:")
                for file in self.current_scenario.files:
                    output.append(f"  - {file.path}")
                output.append("\nCLI History:")
                output.extend(self.current_scenario.cli_history[:5])
                output.append("=" * 80)
            
            rendered = "\n".join(output)
            
            if self.render_mode == "human":
                print(rendered)
            return rendered
        
        return None
    
    def close(self):
        """Clean up resources."""
        if self.sandbox:
            # Sandbox is cleaned up via context manager
            self.sandbox = None

