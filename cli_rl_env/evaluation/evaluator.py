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
        
        # Create environment
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
        commands_executed = []
        
        # Add system prompt
        system_prompt = self._create_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
        
        # Add task description
        task_prompt = self._create_task_prompt(obs)
        messages.append({"role": "user", "content": task_prompt})
        
        # Interaction loop
        done = False
        success = False
        num_commands = 0
        verification_results = {}
        
        while not done and num_commands < self.max_commands:
            try:
                # Get model response
                response = self._call_model(messages)
                
                # Extract command from response
                command = self._extract_command(response)
                
                if not command:
                    # Model didn't provide a valid command
                    break
                
                commands_executed.append(command)
                num_commands += 1
                
                # Execute command in environment
                obs, reward, terminated, truncated, info = env.step(command)
                done = terminated or truncated
                
                # Add model response and environment feedback to conversation
                messages.append({"role": "assistant", "content": response})
                
                feedback = self._create_feedback_prompt(obs, reward, info)
                messages.append({"role": "user", "content": feedback})
                
                # Check if task is complete
                if done:
                    success = info.get('success', False)
                    verification_results = info.get('verification', {})
                
            except Exception as e:
                if self.verbose:
                    print(f"  Error during evaluation: {e}")
                break
        
        end_time = time.time()
        
        return {
            'scenario_id': scenario_data['id'],
            'success': success,
            'num_commands': num_commands,
            'commands': commands_executed,
            'verification': verification_results,
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
            return response.choices[0].message.content
        except Exception as e:
            if self.verbose:
                print(f"  Error calling model: {e}")
            return ""
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for model."""
        return """You are a CLI expert tasked with solving programming and system administration tasks using command-line tools.

You will be given a task description and access to a sandboxed CLI environment. Your goal is to complete the task by executing appropriate commands.

IMPORTANT RULES:
1. Output ONLY the command to execute, nothing else
2. No explanations, no markdown, no code blocks
3. One command per response
4. Use standard CLI tools: ls, cat, grep, sed, awk, git, etc.
5. When you believe the task is complete, output: DONE

Example responses:
- ls -la
- cat main.py
- sed -i 's/old/new/g' file.txt
- DONE
"""
    
    def _create_task_prompt(self, obs: Dict[str, Any]) -> str:
        """Create initial task prompt.
        
        Args:
            obs: Environment observation
            
        Returns:
            Task prompt string
        """
        prompt = f"""TASK: {obs['task_description']}

AVAILABLE FILES:
"""
        for f in obs['files']:
            prompt += f"  - {f['path']}\n"
        
        if obs.get('cli_history'):
            prompt += f"\nCLI HISTORY:\n"
            for entry in obs['cli_history'][:5]:  # Show first 5
                prompt += f"  {entry}\n"
        
        prompt += "\nWhat is your first command?"
        return prompt
    
    def _create_feedback_prompt(
        self,
        obs: Dict[str, Any],
        reward: float,
        info: Dict[str, Any]
    ) -> str:
        """Create feedback prompt after command execution.
        
        Args:
            obs: Environment observation
            reward: Reward received
            info: Additional info
            
        Returns:
            Feedback prompt
        """
        prompt = ""
        
        # Add command output
        if obs.get('last_output'):
            output = obs['last_output'][:500]  # Truncate long output
            prompt += f"OUTPUT:\n{output}\n\n"
        
        # Add any errors
        if obs.get('error'):
            prompt += f"ERROR: {obs['error']}\n\n"
        
        # Check if complete
        if info.get('success'):
            prompt += "Task appears complete! If you're done, output: DONE\n"
        
        prompt += "What is your next command? (or DONE if complete)"
        return prompt
    
    def _extract_command(self, response: str) -> Optional[str]:
        """Extract command from model response.
        
        Args:
            response: Model response text
            
        Returns:
            Extracted command or None
        """
        if not response:
            return None
        
        # Clean up response
        response = response.strip()
        
        # Check for DONE
        if response.upper() == "DONE":
            return "DONE"
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split('\n')
            # Find first line that's not ``` or language identifier
            for line in lines[1:]:
                if line and not line.startswith("```"):
                    return line.strip()
        
        # Take first line (should be the command)
        first_line = response.split('\n')[0].strip()
        return first_line if first_line else None
    
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

