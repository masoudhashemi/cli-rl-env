"""Training loop example for the CodeEditingEnv.

This script demonstrates:
1. Running multiple episodes
2. Collecting statistics
3. Different difficulty levels
4. Both Python and JavaScript scenarios
"""

import gymnasium as gym
import cli_rl_env
import random


def mock_llm_policy(obs, difficulty):
    """Mock LLM policy that returns random valid commands.
    
    In practice, replace this with your actual LLM inference.
    
    Args:
        obs: Observation from environment
        difficulty: Difficulty level
        
    Returns:
        Action dict with commands and time_estimate
    """
    # Parse observation to understand the task
    task = obs['task_description']
    
    # Simple heuristic: look at files, run tests, try to fix
    commands = [
        "ls -la",
        "cat *.py" if 'python' in task.lower() else "cat *.js",
    ]
    
    # Add test command
    if 'python' in task.lower():
        commands.append("pytest test_*.py -v")
    else:
        commands.append("node test_*.js")
    
    # Estimate time (random for this mock)
    time_estimate = random.uniform(3.0, 10.0)
    
    return {
        "commands": commands,
        "time_estimate": time_estimate
    }


def run_episode(env, episode_num):
    """Run a single episode.
    
    Args:
        env: Environment instance
        episode_num: Episode number for logging
        
    Returns:
        Dict with episode statistics
    """
    obs, info = env.reset()
    
    difficulty = info['difficulty']
    language = info['language']
    
    # Get action from policy
    action = mock_llm_policy(obs, difficulty)
    
    # Execute action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Extract statistics
    stats = {
        'episode': episode_num,
        'reward': reward,
        'difficulty': difficulty,
        'language': language,
        'commands_used': info.get('commands_executed', 0),
        'expected_commands': info.get('expected_commands', 0),
        'actual_time': info.get('actual_time', 0),
        'estimated_time': info.get('estimated_time', 0),
    }
    
    # Test results
    if 'verification_results' in info:
        ver_results = info['verification_results']
        if 'test_results' in ver_results:
            test_res = ver_results['test_results']
            stats['tests_passed'] = test_res.get('passed', 0)
            stats['tests_total'] = test_res.get('total', 0)
            stats['tests_success'] = test_res.get('success', False)
    
    return stats


def main():
    """Run training loop with multiple episodes."""
    print("=" * 80)
    print("CODE EDITING RL TRAINING LOOP")
    print("=" * 80)
    
    # Configuration
    num_episodes = 10
    difficulties = ['easy', 'medium', 'hard', 'very_hard']
    
    # Statistics tracking
    all_stats = []
    
    # Run episodes
    for i in range(num_episodes):
        # Vary difficulty across episodes
        difficulty = random.choice(difficulties)
        language = random.choice(['python', 'javascript'])
        
        # Create environment for this episode
        env = gym.make(
            'CodeEditingEnv-v0',
            difficulty=difficulty,
            language=language,
            seed=None  # Random seed for variety
        )
        
        print(f"\n{'='*80}")
        print(f"Episode {i+1}/{num_episodes} - {difficulty.upper()} {language.upper()}")
        print(f"{'='*80}")
        
        # Run episode
        stats = run_episode(env, i+1)
        all_stats.append(stats)
        
        # Display episode results
        print(f"  Reward: {stats['reward']:.2f}")
        print(f"  Commands: {stats['commands_used']} (expected: {stats['expected_commands']})")
        print(f"  Time: {stats['actual_time']:.2f}s (estimated: {stats['estimated_time']:.2f}s)")
        
        if 'tests_total' in stats:
            print(f"  Tests: {stats['tests_passed']}/{stats['tests_total']} passed")
        
        env.close()
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    avg_reward = sum(s['reward'] for s in all_stats) / len(all_stats)
    max_reward = max(s['reward'] for s in all_stats)
    min_reward = min(s['reward'] for s in all_stats)
    
    print(f"\nReward Statistics:")
    print(f"  Average: {avg_reward:.2f}")
    print(f"  Max: {max_reward:.2f}")
    print(f"  Min: {min_reward:.2f}")
    
    # Success rate (tests passing)
    successful = [s for s in all_stats if s.get('tests_success', False)]
    success_rate = len(successful) / len(all_stats) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    # By difficulty
    print(f"\nBy Difficulty:")
    for diff in difficulties:
        diff_stats = [s for s in all_stats if s['difficulty'] == diff]
        if diff_stats:
            avg_reward_diff = sum(s['reward'] for s in diff_stats) / len(diff_stats)
            print(f"  {diff}: {avg_reward_diff:.2f} avg reward ({len(diff_stats)} episodes)")
    
    # By language
    print(f"\nBy Language:")
    for lang in ['python', 'javascript']:
        lang_stats = [s for s in all_stats if s['language'] == lang]
        if lang_stats:
            avg_reward_lang = sum(s['reward'] for s in lang_stats) / len(lang_stats)
            print(f"  {lang}: {avg_reward_lang:.2f} avg reward ({len(lang_stats)} episodes)")
    
    print("\n" + "=" * 80)
    print("Training loop complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

