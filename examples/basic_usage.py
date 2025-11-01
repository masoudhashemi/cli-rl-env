"""Basic usage example of the CodeEditingEnv.

This script demonstrates:
1. Creating the environment
2. Resetting to get an initial observation
3. Taking a single action
4. Viewing the results
"""

import gymnasium as gym
import cli_rl_env


def main():
    """Run a simple episode with the environment."""
    # Create environment with medium difficulty
    # Disable env checker since observation includes 'files' field for evaluation compatibility
    env = gym.make('CodeEditingEnv-v0', difficulty='medium', language='python', disable_env_checker=True)
    
    # Reset environment and get initial observation
    print("=" * 80)
    print("Resetting environment...")
    print("=" * 80)
    obs, info = env.reset(seed=42)
    
    # Display observation
    print("\n📋 TASK DESCRIPTION:")
    print(obs['task_description'])
    
    print("\n📁 FILE TREE:")
    print(obs['file_tree'])
    
    print("\n💻 CLI HISTORY:")
    print(obs['cli_history'])
    
    print("\n📊 INFO:")
    print(f"  Difficulty: {info['difficulty']}")
    print(f"  Language: {info['language']}")
    print(f"  Expected commands: {info['expected_commands']}")
    print(f"  Scenario type: {info['scenario_type']}")
    
    # Create a sample action (list of standard CLI commands)
    # In practice, this would come from your LLM
    action = {
        "commands": [
            "ls",
            "cat calculator.py",
            "grep -n 'def' calculator.py",
            # To fix bugs, use standard commands like:
            # "sed -i 's/a + b/a - b/g' calculator.py"
            # "echo 'new line' >> file.py"
        ],
        "time_estimate": 5.0  # seconds
    }
    
    print("\n" + "=" * 80)
    print("Executing action...")
    print("=" * 80)
    print(f"Commands: {action['commands']}")
    print(f"Time estimate: {action['time_estimate']}s")
    
    # Take action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n🎯 Reward: {reward:.2f}")
    print(f"✅ Terminated: {terminated}")
    
    if 'execution_results' in info:
        exec_results = info['execution_results']
        print(f"\n⏱️  Execution time: {exec_results['total_time']:.3f}s")
        print(f"✓ All commands successful: {exec_results['all_successful']}")
        
        print("\n📝 Command outputs:")
        for result in exec_results['results']:
            print(f"  $ {result['command']}")
            if result['success']:
                output = result['output'][:200]
                print(f"    {output}...")
            else:
                print(f"    Error: {result['error']}")
    
    if 'verification_results' in info:
        print("\n🔍 VERIFICATION:")
        ver_results = info['verification_results']
        
        if 'test_results' in ver_results:
            test_res = ver_results['test_results']
            print(f"  Tests: {test_res.get('passed', 0)}/{test_res.get('total', 0)} passed")
        
        if 'lint_results' in ver_results:
            lint_res = ver_results['lint_results']
            if not lint_res.get('skipped', False):
                print(f"  Lint errors: {lint_res.get('error_count', 0)}")
    
    if 'reward_breakdown' in info:
        print("\n💰 REWARD BREAKDOWN:")
        reward_info = info['reward_breakdown']
        print(f"  Base reward: {reward_info['base_reward']:.2f}")
        print(f"  Time score: {reward_info['time_score']:.2f}")
        print(f"  Regression score: {reward_info['regression_score']:.2f}")
        if 'breakdown' in reward_info:
            breakdown = reward_info['breakdown']
            print(f"  Time penalty: {breakdown['time_penalty']:.2f}")
            print(f"  Regression penalty: {breakdown['regression_penalty']:.2f}")
        print(f"  Total reward: {reward_info['total_reward']:.2f}")
    
    env.close()
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

