"""Tests for the CodeEditingEnv."""

import pytest
import gymnasium as gym
import cli_rl_env


def test_env_creation():
    """Test that environment can be created."""
    env = gym.make('CodeEditingEnv-v0', difficulty='easy')
    assert env is not None
    env.close()


def test_env_reset():
    """Test environment reset."""
    env = gym.make('CodeEditingEnv-v0', difficulty='easy', language='python')
    obs, info = env.reset(seed=42)
    
    # Check observation structure
    assert 'task_description' in obs
    assert 'file_tree' in obs
    assert 'cli_history' in obs
    
    # Check info
    assert 'difficulty' in info
    assert 'language' in info
    assert info['language'] == 'python'
    
    env.close()


def test_env_step_valid_action():
    """Test environment step with valid action."""
    env = gym.make('CodeEditingEnv-v0', difficulty='easy', language='python')
    obs, info = env.reset(seed=42)
    
    action = {
        "commands": ["ls", "cat calculator.py"],
        "time_estimate": 5.0
    }
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check that episode terminates (single-step env)
    assert terminated
    # truncated can be True or False depending on implementation
    
    # Check that reward is numeric
    assert isinstance(reward, (int, float))
    
    # Check info contains expected keys
    assert 'execution_results' in info or 'error' in info
    
    env.close()


def test_env_step_invalid_action():
    """Test environment step with invalid action."""
    env = gym.make('CodeEditingEnv-v0', difficulty='easy')
    obs, info = env.reset(seed=42)
    
    # Invalid action (missing time_estimate)
    action = {"commands": ["ls"]}
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Should terminate with negative reward
    assert terminated
    assert reward < 0
    assert 'error' in info
    
    env.close()


def test_different_difficulties():
    """Test that different difficulties can be created."""
    difficulties = ['easy', 'medium', 'hard', 'very_hard']
    
    for difficulty in difficulties:
        env = gym.make('CodeEditingEnv-v0', difficulty=difficulty)
        obs, info = env.reset(seed=42)
        assert info['difficulty'] == difficulty
        env.close()


def test_different_languages():
    """Test both Python and JavaScript scenarios."""
    for language in ['python', 'javascript']:
        env = gym.make('CodeEditingEnv-v0', language=language)
        obs, info = env.reset(seed=42)
        assert info['language'] == language
        env.close()


def test_reproducibility():
    """Test that same seed produces same scenario."""
    env1 = gym.make('CodeEditingEnv-v0', difficulty='medium', seed=123)
    obs1, _ = env1.reset(seed=123)
    
    env2 = gym.make('CodeEditingEnv-v0', difficulty='medium', seed=123)
    obs2, _ = env2.reset(seed=123)
    
    # Should get same task description
    assert obs1['task_description'] == obs2['task_description']
    
    env1.close()
    env2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

