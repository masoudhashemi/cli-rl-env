"""CLI RL Environment for training LLMs on code editing tasks."""

from gymnasium.envs.registration import register

from cli_rl_env.environment import CodeEditingEnv

register(
    id='CodeEditingEnv-v0',
    entry_point='cli_rl_env.environment:CodeEditingEnv',
    max_episode_steps=1,
)

__version__ = "0.1.0"
__all__ = ["CodeEditingEnv"]

