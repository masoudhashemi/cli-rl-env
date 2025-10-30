"""
Example training script showing how to use dataset with environment.

This demonstrates the complete flow:
1. Load dataset
2. Create prompt
3. Get LLM generation
4. Parse output
5. Run in environment
6. Use reward for training
"""

import json
import random
import re
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import cli_rl_env


def create_prompt(example):
    """Create a prompt for the LLM from dataset example.
    
    Args:
        example: Dataset example dict
        
    Returns:
        Formatted prompt string
    """
    # Format file information (if available)
    if 'files' in example and example['files']:
        file_info = "\n".join([
            f"- {f['path']} ({len(f['content'])} bytes)"
            for f in example['files']
        ])
    else:
        # For advanced scenarios without file contents
        file_info = "(Files will be available in the environment)"
    
    # Format CLI history (first 5 lines) if available
    if 'cli_history' in example and example['cli_history']:
        cli_history = "\n".join(example['cli_history'][:5])
        if len(example['cli_history']) > 5:
            cli_history += f"\n... ({len(example['cli_history'])-5} more lines)"
    else:
        cli_history = "(No previous commands)"
    
    # Create structured prompt
    prompt = f"""You are a coding assistant that fixes bugs using CLI commands.

TASK: {example['task_description']}

FILES AVAILABLE:
{file_info}

PREVIOUS CLI COMMANDS:
{cli_history}

INSTRUCTIONS:
1. Use ONLY standard CLI commands (cat, grep, sed, echo, awk, etc.)
2. DO NOT use custom commands like write_file or replace_in_file
3. Generate a list of commands to fix the bugs
4. Estimate how long it will take in seconds

EXAMPLES OF VALID COMMANDS:
- cat calculator.py
- grep -n "def add" calculator.py
- sed -i 's/a - b/a + b/g' calculator.py
- echo "new line" >> file.py
- pytest test_calculator.py -v

GENERATE YOUR RESPONSE IN THIS EXACT JSON FORMAT:
{{
  "commands": ["command1", "command2", "command3"],
  "time_estimate": 5.0
}}

RESPONSE:"""
    
    return prompt


def parse_llm_output(llm_response):
    """Parse LLM output into action format.
    
    Args:
        llm_response: Text response from LLM
        
    Returns:
        Dict with 'commands' and 'time_estimate'
        
    Raises:
        ValueError: If parsing fails
    """
    response = llm_response.strip()
    
    # Remove markdown code blocks if present
    if "```json" in response:
        match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            response = match.group(1)
    elif "```" in response:
        match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            response = match.group(1)
    
    # Find JSON in response (in case there's extra text)
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        response = json_match.group(0)
    
    # Parse JSON
    try:
        action = json.loads(response)
        
        # Validate format
        if 'commands' not in action:
            raise ValueError("Missing 'commands' field")
        if 'time_estimate' not in action:
            raise ValueError("Missing 'time_estimate' field")
        
        if not isinstance(action['commands'], list):
            raise ValueError("'commands' must be a list")
        if not isinstance(action['time_estimate'], (int, float)):
            raise ValueError("'time_estimate' must be a number")
        
        return action
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nResponse: {response[:200]}")


class MockLLM:
    """Mock LLM for demonstration purposes.
    
    In practice, replace this with your actual LLM:
    - OpenAI API
    - Local model (transformers, vLLM, etc.)
    - Any other LLM service
    """
    
    def generate(self, prompt):
        """Generate a mock response.
        
        In reality, this would call your LLM inference.
        """
        # This is a mock - it just returns basic exploration commands
        # Your actual LLM would generate commands based on the task
        
        return json.dumps({
            "commands": [
                "ls -la",
                "cat *.py | head -20",
                "grep -n 'def' *.py"
            ],
            "time_estimate": 5.0
        })


def run_training_example():
    """Demonstrate the training flow."""
    
    print("=" * 80)
    print("TRAINING WITH DATASET EXAMPLE")
    print("=" * 80)
    print()
    
    # 1. Load dataset
    print("Step 1: Loading dataset...")
    dataset_path = Path(__file__).parent.parent / 'datasets' / 'sample_dataset' / 'train.json'
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Run: uv run python examples/generate_training_dataset.py")
        return
    
    with open(dataset_path) as f:
        train_data = json.load(f)
    
    print(f"✓ Loaded {len(train_data)} training examples")
    print()
    
    # 2. Select example
    example = random.choice(train_data)
    print(f"Step 2: Selected example")
    print(f"  Difficulty: {example['difficulty']}")
    print(f"  Language: {example['language']}")
    print(f"  Expected commands: {example['expected_commands']}")
    print(f"  Task: {example['task_description'][:100]}...")
    print()
    
    # 3. Create prompt
    print("Step 3: Creating prompt for LLM...")
    prompt = create_prompt(example)
    print(f"✓ Prompt created ({len(prompt)} chars)")
    print(f"  First 200 chars: {prompt[:200]}...")
    print()
    
    # 4. Get LLM generation
    print("Step 4: Getting LLM generation...")
    print("  (Using MockLLM - replace with your actual LLM)")
    
    llm = MockLLM()
    llm_response = llm.generate(prompt)
    
    print(f"✓ LLM responded")
    print(f"  Response: {llm_response}")
    print()
    
    # 5. Parse LLM output
    print("Step 5: Parsing LLM output...")
    try:
        action = parse_llm_output(llm_response)
        print(f"✓ Parsed successfully")
        print(f"  Commands: {action['commands']}")
        print(f"  Time estimate: {action['time_estimate']}s")
    except ValueError as e:
        print(f"✗ Parsing failed: {e}")
        return
    print()
    
    # 6. Run in environment
    print("Step 6: Running in environment...")
    
    # Note: For actual training, you'd need to load the scenario
    # from the dataset into the environment. For this demo,
    # we just use the environment's built-in scenarios.
    
    env = gym.make(
        'CodeEditingEnv-v0',
        difficulty=example.get('difficulty', 'medium'),
        language=example.get('language', 'python')
    )
    
    # Reset environment (gets a random scenario of matching difficulty/language)
    obs, info = env.reset(seed=42)
    
    # Execute action
    try:
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✓ Execution complete")
        print(f"  Reward: {reward:.3f} (0-1 range)")
        print()
        
        # 7. Show detailed results
        print("Step 7: Verification results")
        
        if 'verification_results' in info:
            ver = info['verification_results']
            
            if 'test_results' in ver:
                tests = ver['test_results']
                print(f"  Tests: {tests.get('passed', 0)}/{tests.get('total', 0)} passed")
            
            if 'lint_results' in ver:
                lint = ver['lint_results']
                if not lint.get('skipped'):
                    print(f"  Lint errors: {lint.get('error_count', 0)}")
        
        if 'reward_breakdown' in info:
            breakdown = info['reward_breakdown']
            print(f"\n  Reward breakdown:")
            print(f"    Base: {breakdown['base_reward']:.3f}")
            print(f"    Time score: {breakdown['time_score']:.3f}")
            print(f"    Regression score: {breakdown['regression_score']:.3f}")
        
        print()
        
    except Exception as e:
        print(f"✗ Execution failed: {e}")
        reward = 0.0
    
    env.close()
    
    # 8. Training step
    print("Step 8: Training step")
    print(f"  Using reward {reward:.3f} to update LLM")
    print(f"  (In practice: optimizer.step(), backprop, etc.)")
    print()
    
    # Summary
    print("=" * 80)
    print("TRAINING FLOW COMPLETE")
    print("=" * 80)
    print()
    print("Next steps for real training:")
    print("1. Replace MockLLM with your actual LLM")
    print("2. Implement proper training loop (RL or SFT)")
    print("3. Add logging (wandb, tensorboard)")
    print("4. Save checkpoints")
    print("5. Run validation")
    print()
    print("See TRAINING_GUIDE.md for complete details!")


if __name__ == "__main__":
    run_training_example()

