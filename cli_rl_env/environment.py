"""Main Gymnasium environment for code editing tasks."""

import random
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import subprocess

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
        # Note: 'files' is a list of dicts used by evaluation module - we use Text to be flexible
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
                # Commands failed to execute – still run verification to provide signal
                verification_results = self._run_verification(sandbox)
                success = False
                
                # Calculate normalized reward; on failure, clamp to 0.0
                try:
                    reward_info = self.reward_calculator.calculate_reward(
                        verification_results=verification_results,
                        actual_time=execution_results['total_time'],
                        estimated_time=time_estimate
                    )
                    reward = 0.0
                except Exception:
                    reward_info = {
                        'total_reward': 0.0,
                        'base_reward': 0.0,
                        'time_score': 0.0,
                        'regression_score': 1.0,
                        'actual_time': execution_results.get('total_time', 0.0),
                        'estimated_time': time_estimate,
                        'breakdown': {}
                    }
                    reward = 0.0
                
                info = {
                    'success': success,
                    'execution_results': execution_results,
                    'verification_results': verification_results,
                    'reward_breakdown': reward_info,
                    'commands_executed': len(commands),
                    'expected_commands': self.current_scenario.expected_commands,
                    'actual_time': execution_results['total_time'],
                    'estimated_time': time_estimate,
                    'error': 'Command execution failed'
                }
                return self._create_observation(), reward, True, False, info
            
            # Run verification
            verification_results = self._run_verification(sandbox)
            
            # Determine success from verification results
            success = self._determine_success(verification_results)
            
            # Calculate reward
            reward_info = self.reward_calculator.calculate_reward(
                verification_results=verification_results,
                actual_time=execution_results['total_time'],
                estimated_time=time_estimate
            )
            
            reward = reward_info['total_reward']
            
            # Build info dict
            info = {
                'success': success,
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
            Observation dict (includes 'files' field for evaluation compatibility)
        """
        if self.current_scenario is None:
            base_obs = {
                'task_description': '',
                'file_tree': '',
                'cli_history': ''
            }
            # Add files field for evaluation module compatibility
            # This field is not in observation_space but is used by evaluator
            base_obs['files'] = []
            return base_obs
        
        # Create file tree representation
        file_tree = "Files:\n"
        for file in self.current_scenario.files:
            file_tree += f"  - {file.path} ({len(file.content)} bytes)\n"
        
        # Format CLI history
        cli_history = "\n".join(self.current_scenario.cli_history)
        
        # Prepare files list for external use (e.g., evaluation)
        files_list = [{'path': f.path, 'content': f.content} for f in self.current_scenario.files]
        
        # Base observation matching observation_space
        base_obs = {
            'task_description': self.current_scenario.task_description,
            'file_tree': file_tree,
            'cli_history': cli_history,
        }
        
        # Add files field for evaluation module compatibility
        # This field is not in observation_space but is used by evaluator
        # Gymnasium's passive env checker may complain, but this is necessary for evaluation
        base_obs['files'] = files_list
        
        return base_obs
    
    def _run_verification(self, sandbox: Sandbox) -> Dict[str, Any]:
        """Run all verification rules.
        
        Args:
            sandbox: Sandbox with executed commands
            
        Returns:
            Dict with all verification results
        """
        results = {}
        
        # Run tests (highest priority verification)
        test_files = [f for f in self.current_scenario.files if f.is_test]
        if test_files:
            test_runner = TestRunner(
                sandbox_path=sandbox.get_sandbox_path(),
                language=self.current_scenario.language
            )
            test_results = test_runner.run_tests(test_files[0].path)
            results['test_results'] = test_results
        
        # Run linting only on actual code files
        code_files = [f for f in self.current_scenario.files 
                     if not f.is_test and self._should_lint_file(f.path)]
        if code_files:
            linter = Linter(
                sandbox_path=sandbox.get_sandbox_path(),
                language=self.current_scenario.language
            )
            lint_results = linter.lint_file(code_files[0].path, strict=False)
            results['lint_results'] = lint_results
        
        # Run text matching rules
        text_matcher = TextMatcher(sandbox_path=sandbox.get_sandbox_path())
        text_match_results = []
        for rule in self.current_scenario.verification_rules:
            if rule.type == 'text_match':
                # Skip rules without expected pattern (dataset bug)
                if not rule.expected:
                    continue
                result = text_matcher.verify_pattern(
                    filepath=rule.target,
                    pattern=rule.expected,
                    should_exist=True
                )
                text_match_results.append(result)
        
        if text_match_results:
            results['text_match_results'] = text_match_results
        
        # Permissions verification (heuristic expectations)
        perm_verification = self._verify_permissions(sandbox)
        if perm_verification and perm_verification.get('has_expectations', False):
            results['permissions_verification'] = perm_verification
        
        # Git verification (commit expectations)
        git_verification = self._verify_git(sandbox)
        if git_verification and git_verification.get('has_expectations', False):
            results['git_verification'] = git_verification
        
        # Add execution-based verification (always runs)
        # This ensures EVERY task has some verification
        try:
            execution_verification = self._verify_execution(sandbox)
            if execution_verification:
                results['execution_verification'] = execution_verification
        except Exception as e:
            # Log error but don't crash
            results['execution_verification'] = {
                'error': f"Verification failed: {str(e)}",
                'success': False
            }
        
        return results
    
    def _verify_permissions(self, sandbox: Sandbox) -> Dict[str, Any]:
        """Verify file permissions based on heuristic expectations.
        
        Expectations:
        - Executable bit for owner should be set for shell scripts (*.sh) or files with a shebang.
        - README files should be read-only when task mentions 'read-only'.
        
        Returns a dict with per-file results and overall success. If no
        expectations are detected, returns a dict with has_expectations=False.
        """
        try:
            sandbox_path = Path(sandbox.get_sandbox_path())
            desc = (self.current_scenario.task_description or '').lower()
            readonly_hint = ('read-only' in desc) or ('readonly' in desc)
            
            expected_exec = []
            expected_readonly = []
            
            # Determine expectations from file names and contents
            for f in self.current_scenario.files:
                rel = f.path
                p = sandbox_path / rel
                name_lower = rel.lower()
                content = f.content
                has_shebang = content.startswith('#!')
                
                # Executable expectation: *.sh or shebang
                if name_lower.endswith('.sh') or has_shebang:
                    expected_exec.append(rel)
                
                # README read-only expectation if task mentions it
                if readonly_hint and ('readme' in name_lower):
                    expected_readonly.append(rel)
            
            has_expectations = bool(expected_exec or expected_readonly)
            result = {
                'expected_exec': expected_exec,
                'expected_readonly': expected_readonly,
                'exec_ok': [],
                'exec_fail': [],
                'readonly_ok': [],
                'readonly_fail': [],
                'has_expectations': has_expectations,
                'success': True
            }
            
            if not has_expectations:
                return result
            
            # Evaluate expectations against current filesystem
            for rel in expected_exec:
                path = sandbox_path / rel
                if not path.exists():
                    result['exec_fail'].append(rel)
                    continue
                mode = path.stat().st_mode & 0o777
                # Consider any exec bit sufficient; prefer owner exec
                is_exec = bool(mode & 0o111)
                if is_exec:
                    result['exec_ok'].append(rel)
                else:
                    result['exec_fail'].append(rel)
            
            for rel in expected_readonly:
                path = sandbox_path / rel
                if not path.exists():
                    result['readonly_fail'].append(rel)
                    continue
                mode = path.stat().st_mode & 0o777
                no_write = (mode & 0o222) == 0
                if no_write:
                    result['readonly_ok'].append(rel)
                else:
                    result['readonly_fail'].append(rel)
            
            result['success'] = (len(result['exec_fail']) == 0 and len(result['readonly_fail']) == 0)
            return result
        except Exception as e:
            return {
                'has_expectations': False,
                'error': str(e),
                'success': False
            }
    
    def _verify_git(self, sandbox: Sandbox) -> Dict[str, Any]:
        """Verify git state (inside work tree and commit count).
        
        Heuristics for expectations:
        - If scenario type is 'git' (metadata) or task mentions 'git'/'commit', expectations exist.
        - If task mentions 'two commits'/'2 commits'/'second commit', require >=2 commits; else >=1.
        """
        try:
            sandbox_path = Path(sandbox.get_sandbox_path())
            meta = getattr(self.current_scenario, 'metadata', {}) or {}
            scenario_type = meta.get('scenario_type', '').lower()
            desc = (self.current_scenario.task_description or '').lower()
            has_git_expectation = ('git' in scenario_type) or ('git' in desc) or ('commit' in desc)
            
            # If no expectation, return quickly
            result: Dict[str, Any] = {
                'has_expectations': has_git_expectation,
                'inside_work_tree': False,
                'commit_count': 0,
                'log_excerpt': '',
                'success': False
            }
            if not has_git_expectation:
                return result
            
            # Check if inside a work tree
            rp = subprocess.run(
                ['git', 'rev-parse', '--is-inside-work-tree'],
                cwd=str(sandbox_path),
                capture_output=True,
                text=True,
                timeout=5
            )
            inside = rp.returncode == 0 and rp.stdout.strip() == 'true'
            result['inside_work_tree'] = inside
            if not inside:
                result['success'] = False
                return result
            
            # Count commits on HEAD
            rc = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD'],
                cwd=str(sandbox_path),
                capture_output=True,
                text=True,
                timeout=5
            )
            if rc.returncode == 0:
                try:
                    count = int(rc.stdout.strip())
                except ValueError:
                    count = 0
            else:
                count = 0
            result['commit_count'] = count
            
            # Optional: log excerpt
            lg = subprocess.run(
                ['git', 'log', '--oneline', '-n', '3'],
                cwd=str(sandbox_path),
                capture_output=True,
                text=True,
                timeout=5
            )
            if lg.returncode == 0:
                result['log_excerpt'] = lg.stdout.strip()
            
            # Determine required commits
            requires_two = any(token in desc for token in ['two commit', '2 commit', 'second commit'])
            required = 2 if requires_two else 1
            result['required_commits'] = required
            result['success'] = count >= required
            return result
        except Exception as e:
            return {
                'has_expectations': False,
                'error': str(e),
                'success': False
            }
    
    def _verify_execution(self, sandbox: Sandbox) -> Dict[str, Any]:
        """Verify execution by checking if files were actually modified.
        
        This provides a baseline verification for all tasks, ensuring
        that something actually happened.
        
        Args:
            sandbox: Sandbox with executed commands
            
        Returns:
            Dict with execution verification results
        """
        verification = {
            'files_modified': [],
            'files_created': [],
            'files_deleted': [],
            'dirs_created': [],
            'dirs_deleted': [],
            'success': False
        }
        
        try:
            sandbox_path = Path(sandbox.get_sandbox_path())

            # Build initial snapshot of files and directories from scenario
            initial_files = {Path(f.path).as_posix() for f in self.current_scenario.files}
            initial_dirs = set()
            for f in self.current_scenario.files:
                p = Path(f.path)
                for parent in p.parents:
                    if str(parent) == '.':
                        break
                    initial_dirs.add(parent.as_posix())
            
            # Check each file in the scenario
            for file_obj in self.current_scenario.files:
                file_path = sandbox_path / file_obj.path
                
                if file_path.exists():
                    # File exists - check if it was modified
                    try:
                        current_content = file_path.read_text()
                        if current_content != file_obj.content:
                            verification['files_modified'].append(file_obj.path)
                    except Exception:
                        # File might be binary or unreadable
                        pass
                else:
                    # File was deleted
                    verification['files_deleted'].append(file_obj.path)
            
            # Check for new files/directories created (recursive)
            for item in sandbox_path.rglob('*'):
                rel = item.relative_to(sandbox_path).as_posix()
                if item.is_file():
                    if rel not in initial_files:
                        verification['files_created'].append(rel)
                elif item.is_dir():
                    if rel not in initial_dirs and rel != '.':
                        verification['dirs_created'].append(rel)

            # Check for deleted directories (present initially but missing now)
            for d in initial_dirs:
                if not (sandbox_path / d).exists():
                    verification['dirs_deleted'].append(d)
            
            # Consider it successful if ANY file was modified or created
            verification['success'] = bool(
                verification['files_modified'] or 
                verification['files_created'] or
                verification['files_deleted'] or
                verification['dirs_created'] or
                verification['dirs_deleted']
            )
        except Exception as e:
            # If verification fails, mark as unsuccessful but don't crash
            verification['error'] = str(e)
            verification['success'] = False
        
        return verification
    
    def _should_lint_file(self, filepath: str) -> bool:
        """Check if file should be linted based on extension.
        
        Args:
            filepath: Path to file
            
        Returns:
            True if file should be linted
        """
        # Only lint actual code files, not config/data/shell files
        code_extensions = {
            'python': {'.py'},
            'javascript': {'.js', '.jsx', '.ts', '.tsx'}
        }
        
        ext = filepath[filepath.rfind('.'):] if '.' in filepath else ''
        language_exts = code_extensions.get(self.current_scenario.language, set())
        
        return ext in language_exts
    
    def _determine_success(self, verification_results: Dict[str, Any]) -> bool:
        """Determine if task was successful based on verification results.
        
        Args:
            verification_results: Dict with test/lint/text match results
            
        Returns:
            True if task completed successfully
        """
        # Priority 1: Test results (most important)
        if 'test_results' in verification_results:
            test_res = verification_results['test_results']
            # Tests must pass
            if not test_res.get('success', False):
                return False
            # If tests exist, they must all pass
            if test_res.get('total', 0) > 0:
                if test_res.get('failed', 0) > 0:
                    return False
        
        # Priority 2: Text matching (required patterns must exist)
        # But be lenient with broken verification rules
        if 'text_match_results' in verification_results:
            text_matches = verification_results['text_match_results']
            if isinstance(text_matches, list):
                # Filter out broken rules (directory targets, etc.)
                valid_matches = [m for m in text_matches 
                               if not ('error' in m and 'directory' in m.get('error', '').lower())]
                
                # If all rules are broken, consider it a pass (dataset bug)
                if valid_matches:
                    # All valid text match rules must pass
                    if not all(m.get('success', False) for m in valid_matches):
                        return False
            elif isinstance(text_matches, dict):
                # Skip if it's a directory error (dataset bug)
                if not ('error' in text_matches and 'directory' in text_matches.get('error', '').lower()):
                    if not text_matches.get('success', False):
                        return False
        
        # Priority 3: Linting (optional, but no critical errors)
        if 'lint_results' in verification_results:
            lint_res = verification_results['lint_results']
            # Allow linting to be skipped or have minor errors
            # Only fail if there are many errors (>10)
            if not lint_res.get('skipped', False):
                error_count = lint_res.get('error_count', 0)
                if error_count > 10:
                    return False
        
        # Priority 3.5: Permissions verification (if expectations exist)
        if 'permissions_verification' in verification_results:
            perm = verification_results['permissions_verification']
            if perm.get('has_expectations', False) and not perm.get('success', False):
                return False
        
        # Priority 3.7: Git verification (if expectations exist)
        if 'git_verification' in verification_results:
            gv = verification_results['git_verification']
            if gv.get('has_expectations', False) and not gv.get('success', False):
                return False
        
        # Check if we have meaningful verification
        has_tests = 'test_results' in verification_results
        has_valid_text_match = False
        has_permissions = False
        has_git = False
        if 'text_match_results' in verification_results:
            text_matches = verification_results['text_match_results']
            if isinstance(text_matches, list):
                # Check if there are any valid (non-directory-error) rules
                has_valid_text_match = any(
                    not ('error' in m and 'directory' in m.get('error', '').lower())
                    for m in text_matches
                )
            elif isinstance(text_matches, dict):
                has_valid_text_match = not ('error' in text_matches and 'directory' in text_matches.get('error', '').lower())
        if 'permissions_verification' in verification_results:
            perm = verification_results['permissions_verification']
            has_permissions = perm.get('has_expectations', False)
        if 'git_verification' in verification_results:
            gv = verification_results['git_verification']
            has_git = gv.get('has_expectations', False)
        
        # Priority 4: Execution verification (baseline - always present)
        # If no other verification exists, use execution verification
        has_execution_verification = 'execution_verification' in verification_results
        if has_execution_verification:
            exec_verify = verification_results['execution_verification']
            # If we have no other verification, execution verification determines success
            if not has_tests and not has_valid_text_match and not has_permissions and not has_git:
                return exec_verify.get('success', False)
        
        # If we have meaningful verification and all passed, success
        if has_tests:
            # If tests exist, they determine success
            return True
        elif has_valid_text_match:
            # If valid text match rules exist, they passed (we didn't return False above)
            return True
        elif has_permissions:
            # Permissions expectations exist and were met (otherwise we'd have returned False above)
            return True
        elif has_git:
            # Git expectations exist and were met
            return True
        else:
            # No meaningful verification at all - this shouldn't happen now
            # but keep lenient fallback
            return True
    
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

