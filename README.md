# CLI RL Environment

A Gymnasium-based reinforcement learning environment for training large language models (LLMs) to explore, debug, and edit code files. The environment provides isolated sandbox execution, automatic verification, and reward signals based on task correctness and time estimation accuracy.

## Features

- **Model-Agnostic Interface**: Works with any LLM via a simple JSON action format
- **Safe Execution**: Isolated sandbox environments with command whitelisting
- **Multi-Language Support**: Python and JavaScript scenarios with realistic bugs
- **Difficulty Levels**: Easy, Medium, Hard, and Very Hard scenarios
- **Comprehensive Verification**: Unit tests, linting, and text matching
- **Reward Shaping**: Base reward + time penalties + regression penalties
- **Repeatable Episodes**: Each episode runs in isolation with fresh state
- **🆕 High-Diversity Datasets**: 70-80% command coverage with diverse scenario generator
- **🆕 Diversity Analysis**: Built-in tools to measure and ensure command diversity

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd cli-rl-env

# Install dependencies
pip install -e .

# Or install with uv
uv pip install -e .
```

### Requirements

- Python 3.8+
- pytest (for running tests)
- pylint/flake8 (optional, for linting)
- Node.js (optional, for JavaScript scenarios)

## Quick Start

```python
import gymnasium as gym
import cli_rl_env

# Create environment
env = gym.make('CodeEditingEnv-v0', difficulty='medium')

# Reset and get initial observation
obs, info = env.reset()

# Observation contains:
# - obs['task_description']: Natural language task
# - obs['file_tree']: Files in the project
# - obs['cli_history']: Simulated CLI history

# Create action (normally from your LLM)
action = {
    "commands": [
        "ls",
        "cat main.py",
        "sed -i 's/bug/fix/g' main.py"  # Standard sed command
    ],
    "time_estimate": 5.0  # seconds
}

# Execute action
obs, reward, terminated, truncated, info = env.step(action)

# Results in info:
# - info['verification_results']: Test/lint results
# - info['reward_breakdown']: Detailed reward components
# - info['execution_results']: Command execution details
```

## Environment Details

### Observation Space

The environment provides a dictionary observation:

- **task_description**: Natural language description of the debugging/editing task
- **file_tree**: List of files in the project with sizes
- **cli_history**: Simulated command history showing prior exploration

### Action Space

Actions are dictionaries with two fields:

- **commands**: List of command strings to execute
- **time_estimate**: Estimated time (in seconds) to complete the task

### Supported Commands

**Standard CLI/Bash commands only** - no custom commands!

**File Viewing:**
- `cat <file>`, `head <file>`, `tail <file>`, `less <file>`

**File System:**
- `ls [options]`, `find <pattern>`, `tree`, `file`, `stat`

**Text Search/Processing:**
- `grep <pattern> <file>` - Search in files
- `sed 's/old/new/g' <file>` - Stream editor for text replacement
- `awk`, `cut`, `tr`, `sort`, `uniq`, `wc`

**File Operations:**
- `cp`, `mv`, `rm`, `mkdir`, `touch`, `chmod`

**Text Editing (via shell):**
- `echo "content" > file.txt` - Write to file
- `echo "content" >> file.txt` - Append to file
- `cat << 'EOF' > file.txt ... EOF` - Multi-line write
- `sed -i 's/old/new/g' file.txt` - In-place replacement

**Navigation:**
- `cd <dir>`, `pwd`

**Git:**
- All `git` commands

**Testing/Execution:**
- `python`, `python3`, `pytest`, `node`, `npm`

### Reward Structure

**All rewards are normalized to the [0, 1] range** for RL training.

The reward is calculated as:

```
total_reward = base_reward × (1 - time_penalty_weight × time_penalty) × (1 - regression_weight × regression)
```

- **Base Reward (0-1)**: Based on verification results
  - Test pass rate (weight: 0.7)
  - Linting errors (weight: 0.2)
  - Text pattern matching (weight: 0.1)

- **Time Score (0-1)**: Reward for time estimation accuracy
  - 1.0 if within estimate
  - Decreases if over estimate (1/time_ratio)

- **Regression Score (0-1)**: Penalty for breaking tests
  - 1.0 if no tests broke
  - Decreases proportionally to tests broken

### Difficulty Levels

| Level | Commands | Bug Count | Description |
|-------|----------|-----------|-------------|
| Easy | 1-2 | 1 | Single, straightforward bug |
| Medium | 3-5 | 2 | Multiple related issues |
| Hard | 6-10 | 3 | Complex debugging required |
| Very Hard | 10+ | 4+ | Multiple interrelated bugs |

## Examples

### Basic Usage

```bash
python examples/basic_usage.py
```

This demonstrates a single episode with detailed output of observations, actions, and rewards.

### Training Loop

```bash
python examples/training_loop.py
```

This runs multiple episodes across different difficulties and languages, collecting statistics.

### Custom Training Loop

```python
import gymnasium as gym
import cli_rl_env

env = gym.make('CodeEditingEnv-v0', difficulty='hard', language='python')

for episode in range(100):
    obs, info = env.reset()
    
    # Get action from your LLM
    action = your_llm.generate_action(
        task=obs['task_description'],
        files=obs['file_tree'],
        history=obs['cli_history']
    )
    
    # Execute
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Log/store for training
    your_logger.log(reward, info)
```

## Scenario Types

### Python Scenarios

1. **Calculator**: Basic arithmetic operations with bugs
2. **Data Processor**: List/array processing functions
3. **String Utils**: String manipulation utilities
4. **Algorithms**: Classic algorithms (search, sort, recursion)

### JavaScript Scenarios

1. **Utils**: Basic utility functions
2. **Array Operations**: Array manipulation methods
3. **Validators**: Input validation functions

## Verification Methods

The environment supports multiple verification types:

1. **Unit Tests**: Automatic test execution with pytest/node
2. **Linting**: Code quality checks (pylint/flake8 for Python, syntax check for JS)
3. **Text Matching**: Verify specific strings exist/removed
4. **Execution**: Check exit codes and output

## Configuration

```python
env = gym.make(
    'CodeEditingEnv-v0',
    difficulty='medium',        # 'easy', 'medium', 'hard', 'very_hard'
    language='python',          # 'python', 'javascript', or None for random
    seed=42,                    # Random seed for reproducibility
    max_commands=50,            # Maximum commands per episode
    render_mode='human'         # 'human', 'ansi', or None
)
```

## Safety Features

- **Sandbox Isolation**: Each episode runs in a temporary directory
- **Command Whitelisting**: Only safe commands are allowed
- **Path Restrictions**: No absolute paths or directory traversal
- **Resource Limits**: Timeouts on command execution
- **Clean Isolation**: Complete cleanup after each episode

## Architecture

```
cli-rl-env/
├── cli_rl_env/
│   ├── environment.py           # Main Gymnasium environment
│   ├── scenario_generator/      # Generate code scenarios
│   │   ├── base.py
│   │   ├── python_generator.py
│   │   ├── javascript_generator.py
│   │   ├── bug_injector.py
│   │   └── prompt_generator.py
│   ├── executor/                 # Safe command execution
│   │   ├── sandbox.py
│   │   ├── command_parser.py
│   │   └── cli_simulator.py
│   ├── verifier/                 # Verification systems
│   │   ├── test_runner.py
│   │   ├── text_matcher.py
│   │   └── linter.py
│   └── reward/                   # Reward calculation
│       └── calculator.py
├── examples/                     # Example scripts
└── tests/                        # Test suite
```

## Dataset Generation

### Generate Diverse Training Data

For RL training, you'll want datasets with high command diversity:

```python
from cli_rl_env.prompt_dataset_generator import PromptDatasetGenerator

# Generate balanced diverse dataset
generator = PromptDatasetGenerator(seed=42)
dataset = generator.generate_balanced_diverse_dataset(
    num_prompts=1000,
    diverse_scenario_ratio=0.7,  # 70% command-focused scenarios
    output_file="datasets/train.json"
)

# Split into train/val/test
generator.save_dataset_splits(
    dataset,
    output_dir="datasets/final",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

### Analyze Dataset Diversity

```python
from cli_rl_env.utils.diversity_analyzer import analyze_dataset_diversity

# Get detailed diversity report
report = analyze_dataset_diversity("datasets/final/train.json")

# Check command coverage
print(f"Coverage: {report['command_coverage']['percentage']:.1f}%")
print(f"Commands: {report['command_coverage']['used_commands']}/30")
```

**Expected Results** with `diverse_scenario_ratio=0.7`:
- 📊 70-80% command coverage (21-25 / 30 commands)
- 🎯 15-20 different scenario types
- ✅ 100% coverage of text processing, file search, and git commands

See `QUICKSTART_DIVERSE.md` for detailed guide and `DIVERSITY_IMPROVEMENTS.md` for technical details.

## Advanced Usage

### Custom Scenarios

You can extend the scenario generators to add custom bug types or project templates:

```python
from cli_rl_env.scenario_generator.python_generator import PythonScenarioGenerator

class CustomGenerator(PythonScenarioGenerator):
    def generate(self, difficulty):
        # Your custom scenario generation
        pass
```

### Custom Reward Function

```python
from cli_rl_env.reward.calculator import RewardCalculator

class CustomRewardCalculator(RewardCalculator):
    def calculate_reward(self, verification_results, actual_time, estimated_time):
        # Your custom reward logic
        pass
```

### Integrating with RL Libraries

The environment works with standard RL libraries:

```python
# Stable-Baselines3 example (requires wrapper for Dict observation)
from stable_baselines3 import PPO

# Your custom wrapper to handle Dict observations
env = YourWrapper(gym.make('CodeEditingEnv-v0'))
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Logging and Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

env = gym.make('CodeEditingEnv-v0', render_mode='human')
```

## Contributing

Contributions are welcome! Areas for improvement:

- Additional scenario types and bug patterns
- More programming languages
- Advanced verification methods
- Performance optimizations
- Better error messages

## License

MIT License - see LICENSE file for details

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{cli_rl_env,
  title={CLI RL Environment: A Gymnasium Environment for Code Editing},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/cli-rl-env}
}
```

## Troubleshooting

### Tests not running

Ensure pytest is installed:
```bash
pip install pytest pytest-timeout
```

### JavaScript scenarios failing

Install Node.js:
```bash
# macOS
brew install node

# Ubuntu
sudo apt install nodejs
```

### Linting errors

Install linting tools:
```bash
pip install pylint flake8
```

## Contact

For questions or issues, please open an issue on GitHub.

