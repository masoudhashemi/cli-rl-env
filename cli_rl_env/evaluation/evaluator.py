"""Model evaluator using LiteLLM to connect to vLLM hosts."""

import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("Warning: litellm not installed. Install with: pip install litellm")

from cli_rl_env.environment import CodeEditingEnv
from cli_rl_env.evaluation.metrics import EvaluationMetrics


@dataclass
class ModelConfig:
    """Configuration for model connection."""
    model_name: str  # e.g., "openai/gpt-4" or "hosted-vllm/model-name"
    api_base: Optional[str] = None  # vLLM endpoint: "http://localhost:8000/v1"
    api_key: Optional[str] = "EMPTY"  # For vLLM, usually "EMPTY" or "token-abc123"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120


class ModelEvaluator:
    """Evaluate models on CLI tasks using LiteLLM."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        max_commands: int = 50,
        verbose: bool = True
    ):
        """Initialize evaluator.
        
        Args:
            model_config: Model configuration
            max_commands: Maximum commands per episode
            verbose: Print progress messages
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is required for evaluation. "
                "Install with: pip install litellm"
            )
        
        self.model_config = model_config
        self.max_commands = max_commands
        self.verbose = verbose
        self.metrics = EvaluationMetrics()
    
    def evaluate_dataset(
        self,
        dataset_path: str,
        num_examples: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate model on a dataset.
        
        Args:
            dataset_path: Path to JSON dataset file
            num_examples: Number of examples to evaluate (None = all)
            output_path: Path to save results (optional)
            
        Returns:
            Evaluation results dictionary
        """
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        if num_examples:
            dataset = dataset[:num_examples]
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"EVALUATING MODEL ON {len(dataset)} EXAMPLES")
            print(f"{'='*80}")
            print(f"Model: {self.model_config.model_name}")
            if self.model_config.api_base:
                print(f"API Base: {self.model_config.api_base}")
            print(f"Max Commands: {self.max_commands}")
            print(f"{'='*80}\n")
        
        # Evaluate each example
        results = []
        for i, scenario_data in enumerate(dataset, 1):
            if self.verbose:
                print(f"\n[{i}/{len(dataset)}] Evaluating: {scenario_data['id']}")
                print(f"  Difficulty: {scenario_data['difficulty']}")
                print(f"  Type: {scenario_data.get('metadata', {}).get('scenario_type', 'unknown')}")
            
            result = self.evaluate_scenario(scenario_data)
            results.append(result)
            
            # Update metrics
            self.metrics.add_result(result)
            
            if self.verbose:
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                print(f"  Result: {status}")
                print(f"  Commands: {result['num_commands']}/{self.max_commands}")
                if result.get('verification'):
                    print(f"  Verification: {result['verification']}")
        
        # Generate summary
        summary = self.metrics.get_summary()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("EVALUATION SUMMARY")
            print(f"{'='*80}")
            print(f"Total Examples: {summary['total']}")
            print(f"Passed: {summary['passed']} ({summary['pass_rate']:.1f}%)")
            print(f"Failed: {summary['failed']}")
            print(f"Average Commands: {summary['avg_commands']:.1f}")
            print(f"Average Time: {summary['avg_time']:.2f}s")
            print(f"{'='*80}\n")
        
        # Save results if output path provided
        if output_path:
            self._save_results(results, summary, output_path)
            if self.verbose:
                print(f"Results saved to: {output_path}")
        
        return {
            'summary': summary,
            'results': results,
            'model_config': asdict(self.model_config)
        }
    
    def evaluate_scenario(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model on a single scenario.
        
        Args:
            scenario_data: Scenario data from dataset
            
        Returns:
            Result dictionary
        """
        start_time = time.time()
        
        # Create environment directly (without gym.make() wrappers)
        # This bypasses Gymnasium's passive env checker
        env = CodeEditingEnv()
        
        # Convert scenario_data to scenario object (simplified)
        from cli_rl_env.scenario_generator.base import (
            Scenario, FileContent, VerificationRule, DifficultyLevel
        )
        
        # Parse scenario
        files = [
            FileContent(
                path=f['path'],
                content=f['content'],
                is_test=f.get('is_test', False)
            )
            for f in scenario_data['files']
        ]
        
        verification_rules = [
            VerificationRule(
                type=rule['type'],
                target=rule.get('target'),
                expected=rule.get('expected'),
                description=rule.get('description', '')
            )
            for rule in scenario_data['verification_rules']
        ]
        
        scenario = Scenario(
            difficulty=DifficultyLevel[scenario_data['difficulty'].upper()],
            language=scenario_data['language'],
            task_description=scenario_data['task_description'],
            files=files,
            verification_rules=verification_rules,
            expected_commands=scenario_data.get('expected_commands', 10),
            cli_history=scenario_data.get('cli_history', []),
            metadata=scenario_data.get('metadata', {})
        )
        
        # Reset environment with scenario
        obs, info = env.reset(options={'scenario': scenario})
        
        # Initialize conversation history
        messages = []
        
        # Add system prompt
        system_prompt = self._create_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
        
        # Add task description
        task_prompt = self._create_task_prompt(obs)
        messages.append({"role": "user", "content": task_prompt})
        
        # Get model response with all commands and time estimate
        try:
            response = self._call_model(messages)
            
            # Parse JSON response
            action = self._parse_action_response(response)
            
            if self.verbose and action:
                print(f"[PARSED ACTION]")
                print(f"  Commands: {len(action['commands'])}")
                print(f"  Time estimate: {action['time_estimate']}s")
                for i, cmd in enumerate(action['commands'], 1):
                    print(f"    {i}. {cmd}")
            
            if not action:
                # Failed to parse valid action – return a consistent verification stub
                end_time = time.time()
                return {
                    'scenario_id': scenario_data['id'],
                    'success': False,
                    'num_commands': 0,
                    'commands': [],
                    'verification': {
                        'parse_error_verification': {
                            'success': False,
                            'error': 'Failed to parse model response'
                        }
                    },
                    'time_seconds': end_time - start_time,
                    'difficulty': scenario_data['difficulty'],
                    'scenario_type': scenario_data.get('metadata', {}).get('scenario_type', 'unknown'),
                    'error': 'Failed to parse model response'
                }
            
            # Execute all commands in single step
            obs, reward, terminated, truncated, info = env.step(action)
            
            success = info.get('success', False)
            verification_results = info.get('verification_results', {})
            commands_executed = action['commands']
            num_commands = len(commands_executed)
            
            # Print detailed verification results if verbose
            if self.verbose:
                self._print_verification_details(verification_results, success)
            
        except Exception as e:
            if self.verbose:
                print(f"  Error during evaluation: {e}")
            end_time = time.time()
            return {
                'scenario_id': scenario_data['id'],
                'success': False,
                'num_commands': 0,
                'commands': [],
                'verification': {},
                'time_seconds': end_time - start_time,
                'difficulty': scenario_data['difficulty'],
                'scenario_type': scenario_data.get('metadata', {}).get('scenario_type', 'unknown'),
                'error': str(e)
            }
        
        end_time = time.time()
        
        return {
            'scenario_id': scenario_data['id'],
            'success': success,
            'num_commands': num_commands,
            'commands': commands_executed,
            'verification': verification_results,
            'reward_breakdown': info.get('reward_breakdown'),
            'execution_results': info.get('execution_results'),
            'time_seconds': end_time - start_time,
            'difficulty': scenario_data['difficulty'],
            'scenario_type': scenario_data.get('metadata', {}).get('scenario_type', 'unknown')
        }
    
    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        """Call model using LiteLLM.
        
        Args:
            messages: Conversation history
            
        Returns:
            Model response text
        """
        try:
            response = completion(
                model=self.model_config.model_name,
                messages=messages,
                api_base=self.model_config.api_base,
                api_key=self.model_config.api_key,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
                timeout=self.model_config.timeout
            )
            
            if self.verbose:
                # Extract task description from messages for context
                task_desc = None
                for msg in messages:
                    if msg['role'] == 'user' and 'TASK:' in msg['content']:
                        # Extract just the task line
                        lines = msg['content'].split('\n')
                        for line in lines:
                            if line.startswith('TASK:'):
                                task_desc = line.replace('TASK:', '').strip()
                                break
                        break
                
                if task_desc:
                    print(f"\n[TASK] {task_desc}")
                print(f"[LLM RESPONSE] {response.choices[0].message.content}\n")
            
            return response.choices[0].message.content
        except Exception as e:
            if self.verbose:
                print(f"  Error calling model: {e}")
            return ""
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for model (OS-aware guidance) without f-string braces issues."""
        import platform
        os_name = platform.system().lower()
        if 'darwin' in os_name or 'mac' in os_name:
            sed_note = (
                "macOS sed note: Use BSD sed. For in-place edits, prefer `sed -i '' -e 's/old/new/' file`.\n"
                "Do NOT use sed insert/append/change forms (`i`, `a`, `c`) on macOS in a single line — they require complex escaping and usually fail under our constraints.\n"
                "Instead, for line insertions use printf and a temp file, then mv. Examples:\n"
                "- Prepend a line: `printf '# Updated for development\\n' > tmp && cat config.env >> tmp && mv tmp config.env`\n"
                "- Append a line: `printf '# Updated for development\\n' >> config.env`\n"
                "- Multiple replacements: `sed -i '' -e 's/DEBUG=false/DEBUG=true/' -e 's/LOG_LEVEL=info/LOG_LEVEL=debug/' config.env`\n"
            )
        else:
            sed_note = (
                "Linux sed note: GNU sed is available. Use `sed -i 's/old/new/' file` for in-place edits.\n"
                "For multi-line content, prefer multiple single-line replacements or write a temp file with `printf` then `mv`.\n"
            )

        header = (
            "You are a CLI expert tasked with solving programming and system administration tasks using command-line tools.\n\n"
            "You will be given a task description, file contents, and access to a sandboxed CLI environment. Your goal is to complete the task by planning and executing ALL necessary commands IN A SINGLE TURN.\n\n"
            "IMPORTANT: This is SINGLE-TURN execution:\n"
            "- You will see all file contents upfront\n"
            "- You must plan ALL commands at once\n"
            "- Commands execute sequentially, but you won't see intermediate output\n"
            "- Think carefully and include verification steps\n\n"
            "OS-specific guidance:\n"
        )

        git_note = (
            "Git guidance: Set your identity before committing to avoid errors.\n"
            "Run: `git config user.name \"CI Runner\" && git config user.email \"runner@example.com\"`\n"
            "When tasks mention two commits or a 'second commit', make at least two separate commits.\n"
        )

        middle = "\n" + sed_note + git_note + "\n"

        tail = (
            "You must output a JSON object with:\n"
            "1. \"commands\": A list of ALL commands needed to complete the task (in order)\n"
            "2. \"time_estimate\": Your estimate of total execution time in seconds\n\n"
            "AVAILABLE COMMANDS (whitelist - ONLY these commands are allowed):\n"
            "- File viewing: cat, head, tail, less, more, tac\n"
            "- File system: ls, find, tree, file, stat, du, df, realpath\n"
            "- Text processing: grep, sed, awk, cut, tr, sort, uniq, wc, nl, paste, column\n"
            "- File operations: cp, mv, rm, mkdir, touch, chmod, chown, ln, readlink\n"
            "- Text editing: echo, printf, tee\n"
            "- Comparison: diff, cmp, comm\n"
            "- Patching: patch (use with input redirection: patch file.py < changes.patch)\n"
            "- Navigation: cd, pwd\n"
            "- Git: git (all subcommands)\n"
            "- Testing: python, python3, node, pytest, npm, jest, mocha, pip, pip3\n"
            "- Compression: tar, gzip, gunzip, zip, unzip, bzip2, bunzip2\n"
            "- Checksums: md5sum, sha1sum, sha256sum, cksum\n"
            "- Shell: bash, sh\n"
            "- Utilities: seq, date, env, xargs, basename, dirname, which, type, test, expr, bc, jq, shuf\n\n"
            "FORBIDDEN - DO NOT USE THESE:\n"
            "❌ apply_patch (not a real Unix command - use 'patch' instead)\n"
            "❌ heredocs: command <<EOF or command <<'DELIMITER'\n"
            "❌ multi-line strings or commands spanning multiple lines\n"
            "❌ comments as commands: \"# this is a comment\"\n"
            "❌ backticks: `command`\n"
            "❌ command substitution: $(command)\n"
            "❌ semicolons to chain commands: cmd1; cmd2\n"
            "❌ background processes: command &\n"
            "❌ any command not in the whitelist above\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "- Each command must be a SINGLE LINE\n"
            "- For file editing, use sed, awk, or echo with redirection\n"
            "- Use && or || to chain commands, NOT semicolons\n"
            "- Pipes (|), redirections (>, >>), and logical operators (&&, ||) are allowed\n"
            "- For complex edits, use multiple sed commands or create temp files\n\n"
            "STRICTLY FORBIDDEN (enforced):\n"
            "- Do NOT use heredocs (e.g., `<<EOF`) or embed multi-line scripts.\n- Do NOT use backticks or command substitution.\n- Do NOT output comments as commands.\n- Only use whitelisted commands.\n\n"
            "RULES:\n"
            "- Plan the complete solution before outputting\n"
            "- Include investigation, action, and verification commands\n"
            "- Estimate realistic execution time (consider file I/O, test runs, etc.)\n"
            "- Output ONLY valid JSON, no markdown, no explanations\n"
            "- Every command in your list must be executable as-is on a single line\n\n"
            "CORRECT EXAMPLES:\n\n"
            "Example 1 - Basic permissions (file contents provided, no need to cat first):\n"
        )

        return header + middle + tail
    
    def _create_task_prompt(self, obs: Dict[str, Any]) -> str:
        """Create initial task prompt.
        
        Args:
            obs: Environment observation
            
        Returns:
            Task prompt string
        """
        prompt = f"""TASK: {obs['task_description']}

ENVIRONMENT:
  Python: 3.11.3
  Node: 18.x  
  Shell: bash
  Working Directory: /sandbox (all files are in this directory)

AVAILABLE FILES AND CONTENTS:
"""
        # Include file contents so model can see what needs to be fixed
        for f in obs['files']:
            prompt += f"\n--- {f['path']} ---\n"
            content = f['content']
            # Truncate very long files (>2000 chars) to avoid token limits
            if len(content) > 2000:
                prompt += content[:2000] + f"\n... (truncated, {len(content)} total bytes)\n"
            else:
                prompt += content + "\n"
        
        if obs.get('cli_history'):
            prompt += f"\nCLI HISTORY:\n"
            for entry in obs['cli_history'][:5]:  # Show first 5
                prompt += f"  {entry}\n"
        
        prompt += "\n\nProvide your complete solution as a JSON object with all commands and time estimate."
        return prompt
    
    def _parse_action_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse model response into action format.
        
        Args:
            response: Model response text
            
        Returns:
            Action dict with 'commands' and 'time_estimate', or None if invalid
        """
        if not response:
            return None
        
        # Clean up response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split('\n')
            # Find JSON content between ``` markers
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (not line.startswith("```") and json_lines):
                    json_lines.append(line)
            response = '\n'.join(json_lines).strip()
        
        # Try to parse JSON
        try:
            action = json.loads(response)
            
            # Validate structure
            if not isinstance(action, dict):
                if self.verbose:
                    print(f"  Error: Response is not a JSON object")
                return None
            
            if 'commands' not in action:
                if self.verbose:
                    print(f"  Error: Missing 'commands' field in response")
                return None
            
            if 'time_estimate' not in action:
                if self.verbose:
                    print(f"  Error: Missing 'time_estimate' field in response")
                return None
            
            if not isinstance(action['commands'], list):
                if self.verbose:
                    print(f"  Error: 'commands' must be a list")
                return None
            
            if not isinstance(action['time_estimate'], (int, float)):
                if self.verbose:
                    print(f"  Error: 'time_estimate' must be a number")
                return None
            
            return action
            
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"  Error: Invalid JSON in response: {e}")
                print(f"  Response was: {response[:200]}")
            return None
    
    def _print_verification_details(
        self,
        verification_results: Dict[str, Any],
        success: bool
    ):
        """Print detailed verification results for debugging.
        
        Args:
            verification_results: Verification results dict
            success: Overall success flag
        """
        print(f"  [VERIFICATION] Overall Success: {'✓' if success else '✗'}")
        
        # Test results
        if 'test_results' in verification_results:
            test_res = verification_results['test_results']
            print(f"  [TESTS] ", end="")
            if test_res.get('success'):
                passed = test_res.get('passed', 0)
                total = test_res.get('total', 0)
                print(f"✓ Passed {passed}/{total}")
            else:
                passed = test_res.get('passed', 0)
                failed = test_res.get('failed', 0)
                total = test_res.get('total', 0)
                print(f"✗ Failed {failed}/{total} (passed: {passed})")
                if test_res.get('output'):
                    # Show last few lines of test output
                    lines = test_res['output'].split('\n')
                    error_lines = [l for l in lines if 'FAILED' in l or 'ERROR' in l or 'Error' in l]
                    if error_lines:
                        print(f"    Error: {error_lines[0][:100]}")
        
        # Lint results
        if 'lint_results' in verification_results:
            lint_res = verification_results['lint_results']
            if lint_res.get('skipped'):
                print(f"  [LINT] ⊘ Skipped")
            else:
                error_count = lint_res.get('error_count', 0)
                if error_count == 0:
                    print(f"  [LINT] ✓ No errors")
                else:
                    print(f"  [LINT] ⚠ {error_count} errors")
                    if lint_res.get('output'):
                        # Show first error
                        lines = lint_res['output'].split('\n')
                        if lines:
                            print(f"    First: {lines[0][:100]}")
        
        # Text match results
        if 'text_match_results' in verification_results:
            text_matches = verification_results['text_match_results']
            if isinstance(text_matches, list):
                passed = sum(1 for m in text_matches if m.get('success', False))
                total = len(text_matches)
                print(f"  [TEXT_MATCH] {passed}/{total} patterns matched")
                # Show failures
                for i, match in enumerate(text_matches):
                    if not match.get('success', False):
                        error = match.get('error', 'Unknown error')
                        pattern = match.get('pattern', 'N/A')
                        print(f"    ✗ Pattern {i+1}: {error[:80]}")
                        if pattern and pattern != 'N/A':
                            print(f"      Looking for: {pattern[:60]}")
            elif isinstance(text_matches, dict):
                if text_matches.get('success'):
                    print(f"  [TEXT_MATCH] ✓ Pattern matched")
                else:
                    error = text_matches.get('error', 'Unknown error')
                    print(f"  [TEXT_MATCH] ✗ {error[:80]}")

        # Permissions verification
        if 'permissions_verification' in verification_results:
            perm = verification_results['permissions_verification']
            if perm.get('has_expectations', False):
                ok_exec = len(perm.get('exec_ok', []))
                fail_exec = len(perm.get('exec_fail', []))
                ok_ro = len(perm.get('readonly_ok', []))
                fail_ro = len(perm.get('readonly_fail', []))
                status = '✓' if perm.get('success', False) else '✗'
                print(f"  [PERMISSIONS] {status} exec ok/fail: {ok_exec}/{fail_exec}, readonly ok/fail: {ok_ro}/{fail_ro}")
        
        # Execution verification (baseline)
        if 'execution_verification' in verification_results:
            exec_verify = verification_results['execution_verification']
            if exec_verify.get('success'):
                modified = len(exec_verify.get('files_modified', []))
                created = len(exec_verify.get('files_created', []))
                deleted = len(exec_verify.get('files_deleted', []))
                print(f"  [EXECUTION] ✓ Files changed: {modified} modified, {created} created, {deleted} deleted")
            else:
                print(f"  [EXECUTION] ✗ No files were modified or created")
    
    def _save_results(
        self,
        results: List[Dict[str, Any]],
        summary: Dict[str, Any],
        output_path: str
    ):
        """Save evaluation results to file.
        
        Args:
            results: List of result dictionaries
            summary: Summary statistics
            output_path: Path to save results
        """
        output = {
            'model_config': asdict(self.model_config),
            'summary': summary,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

