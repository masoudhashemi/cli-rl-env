"""Generate large training datasets for LLM training.

This script generates diverse, challenging prompts for training your LLM
on code editing tasks.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli_rl_env.prompt_dataset_generator import PromptDatasetGenerator


def main():
    """Generate training dataset."""
    parser = argparse.ArgumentParser(description='Generate LLM training dataset')
    parser.add_argument(
        '--num-prompts',
        type=int,
        default=2000,
        help='Number of prompts to generate (default: 2000)'
    )
    parser.add_argument(
        '--num-advanced',
        type=int,
        default=1000,
        help='Number of advanced prompts to generate (default: 1000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets/code_editing',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--hard-focus',
        action='store_true',
        help='Focus on harder prompts (70%% hard+very_hard)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GENERATING LLM TRAINING DATASET")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Standard prompts: {args.num_prompts}")
    print(f"  Advanced prompts: {args.num_advanced}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Random seed: {args.seed}")
    print(f"  Hard focus: {args.hard_focus}")
    print()
    
    # Initialize generator
    generator = PromptDatasetGenerator(seed=args.seed)
    
    # Set difficulty distribution
    if args.hard_focus:
        # Focus on harder scenarios
        difficulty_dist = {
            'easy': 0.05,
            'medium': 0.15,
            'hard': 0.45,
            'very_hard': 0.35
        }
        print("Using hard-focused difficulty distribution:")
    else:
        # Balanced distribution with bias toward harder
        difficulty_dist = {
            'easy': 0.10,
            'medium': 0.20,
            'hard': 0.40,
            'very_hard': 0.30
        }
        print("Using standard difficulty distribution:")
    
    for diff, ratio in difficulty_dist.items():
        print(f"  {diff}: {ratio * 100:.0f}%")
    print()
    
    # Generate standard scenarios
    print(f"\nGenerating {args.num_prompts} standard scenarios...")
    standard_dataset = generator.generate_dataset(
        num_prompts=args.num_prompts,
        difficulty_distribution=difficulty_dist
    )
    
    # Generate advanced scenarios
    print(f"\nGenerating {args.num_advanced} advanced scenarios...")
    advanced_dataset = generator.generate_advanced_scenarios(
        num_prompts=args.num_advanced
    )
    
    # Combine datasets
    full_dataset = standard_dataset + advanced_dataset
    print(f"\nTotal dataset size: {len(full_dataset)} prompts")
    
    # Save splits
    print(f"\nSaving dataset splits to {args.output_dir}/...")
    generator.save_dataset_splits(
        dataset=full_dataset,
        output_dir=args.output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Print statistics
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    
    print(f"\nDataset location: {args.output_dir}/")
    print(f"  - train.json: Training set")
    print(f"  - val.json: Validation set")
    print(f"  - test.json: Test set")
    print(f"  - stats.json: Dataset statistics")
    
    print("\nNext steps:")
    print("  1. Load the dataset in your training script")
    print("  2. Format prompts for your LLM")
    print("  3. Train with RL using the environment")
    print("  4. Evaluate on the test set")
    
    print("\nExample usage:")
    print("```python")
    print("import json")
    print("with open('datasets/code_editing/train.json') as f:")
    print("    train_data = json.load(f)")
    print("```")


if __name__ == "__main__":
    main()

