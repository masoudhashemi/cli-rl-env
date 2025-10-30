"""Verification script to test the CLI RL Environment installation."""

import sys
import gymnasium as gym
import cli_rl_env


def test_import():
    """Test that the package can be imported."""
    print("✓ Package imported successfully")
    print(f"  Version: {cli_rl_env.__version__}")


def test_env_creation():
    """Test environment creation."""
    try:
        env = gym.make('CodeEditingEnv-v0', difficulty='easy')
        env.close()
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False
    return True


def test_reset():
    """Test environment reset."""
    try:
        env = gym.make('CodeEditingEnv-v0', difficulty='medium', language='python')
        obs, info = env.reset(seed=42)
        
        assert 'task_description' in obs
        assert 'file_tree' in obs
        assert 'cli_history' in obs
        assert 'difficulty' in info
        assert 'language' in info
        
        env.close()
        print("✓ Environment reset works")
        print(f"  Generated task: {obs['task_description'][:60]}...")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        return False
    return True


def test_step():
    """Test environment step."""
    try:
        env = gym.make('CodeEditingEnv-v0', difficulty='easy', language='python')
        obs, info = env.reset(seed=42)
        
        action = {
            "commands": ["ls"],
            "time_estimate": 1.0
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert terminated  # Single-step environment
        assert isinstance(reward, (int, float))
        
        env.close()
        print("✓ Environment step works")
        print(f"  Reward: {reward:.2f}")
    except Exception as e:
        print(f"✗ Step failed: {e}")
        return False
    return True


def test_multiple_difficulties():
    """Test different difficulty levels."""
    try:
        difficulties = ['easy', 'medium', 'hard', 'very_hard']
        for diff in difficulties:
            env = gym.make('CodeEditingEnv-v0', difficulty=diff)
            obs, info = env.reset(seed=42)
            env.close()
        print(f"✓ All difficulty levels work: {', '.join(difficulties)}")
    except Exception as e:
        print(f"✗ Difficulty test failed: {e}")
        return False
    return True


def test_multiple_languages():
    """Test different programming languages."""
    try:
        languages = ['python', 'javascript']
        for lang in languages:
            env = gym.make('CodeEditingEnv-v0', language=lang)
            obs, info = env.reset(seed=42)
            env.close()
        print(f"✓ All languages work: {', '.join(languages)}")
    except Exception as e:
        print(f"✗ Language test failed: {e}")
        return False
    return True


def main():
    """Run all verification tests."""
    print("=" * 80)
    print("CLI RL Environment - Installation Verification")
    print("=" * 80)
    print()
    
    tests = [
        ("Import", test_import),
        ("Environment Creation", test_env_creation),
        ("Reset", test_reset),
        ("Step", test_step),
        ("Multiple Difficulties", test_multiple_difficulties),
        ("Multiple Languages", test_multiple_languages),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            result = test_func()
            if result is None:
                result = True
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 All tests passed! The environment is ready to use.")
        print("\nNext steps:")
        print("  1. Run: uv run python examples/basic_usage.py")
        print("  2. Run: uv run python examples/training_loop.py")
        print("  3. Read: QUICKSTART.md")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

