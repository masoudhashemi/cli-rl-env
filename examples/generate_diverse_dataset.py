"""Example: Generate a balanced, command-diverse training dataset.

This script demonstrates how to:
1. Generate a dataset with high command diversity
2. Analyze the diversity of the generated dataset
3. Ensure all CLI commands are well-represented
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli_rl_env.prompt_dataset_generator import PromptDatasetGenerator
from cli_rl_env.utils.diversity_analyzer import analyze_dataset_diversity


def main():
    """Generate and analyze a diverse dataset."""
    
    print("="*80)
    print("GENERATING COMMAND-DIVERSE TRAINING DATASET")
    print("="*80)
    
    # Initialize generator with seed for reproducibility
    generator = PromptDatasetGenerator(seed=42)
    
    # Configuration
    output_dir = "datasets/diverse_dataset"
    num_train = 800
    num_val = 100
    num_test = 100
    total = num_train + num_val + num_test
    
    # Generate balanced diverse dataset
    # This uses 60% diverse scenarios that focus on different commands
    print(f"\nGenerating {total} training examples...")
    print("  - 60% diverse scenarios (covering all command categories)")
    print("  - 40% standard scenarios (bug fixes, refactoring, etc.)")
    print()
    
    dataset = generator.generate_balanced_diverse_dataset(
        num_prompts=total,
        difficulty_distribution={
            'easy': 0.1,      # 10%
            'medium': 0.2,    # 20%
            'hard': 0.4,      # 40%
            'very_hard': 0.3  # 30%
        },
        diverse_scenario_ratio=0.6,  # 60% diverse scenarios
        output_file=f"{output_dir}/full_dataset.json"
    )
    
    # Split into train/val/test
    print(f"\nSplitting dataset...")
    generator.save_dataset_splits(
        dataset,
        output_dir=output_dir,
        train_ratio=num_train/total,
        val_ratio=num_val/total,
        test_ratio=num_test/total
    )
    
    # Analyze diversity
    print("\n" + "="*80)
    print("DIVERSITY ANALYSIS")
    print("="*80)
    
    # Analyze training set
    print("\n📊 Training Set Diversity:")
    print("-" * 80)
    train_report = analyze_dataset_diversity(f"{output_dir}/train.json")
    
    # Check if diversity goals are met
    coverage = train_report['command_coverage']['percentage']
    missing = train_report['command_coverage']['missing_commands']
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if coverage >= 90:
        print("✅ EXCELLENT: Command coverage is >= 90%")
    elif coverage >= 75:
        print("✓ GOOD: Command coverage is >= 75%")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Command coverage is < 75%")
    
    if missing:
        print(f"\n⚠️  Missing commands: {', '.join(missing)}")
        print("Consider adding specific scenarios for these commands.")
    else:
        print("\n✅ All commands are represented in the dataset!")
    
    print(f"\nDataset saved to: {output_dir}/")
    print(f"  - train.json: {num_train} examples")
    print(f"  - val.json: {num_val} examples")
    print(f"  - test.json: {num_test} examples")
    print(f"  - full_dataset.json: {total} examples")
    print(f"  - stats.json: Dataset statistics")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review the diversity report above")
    print("2. If coverage is low, increase diverse_scenario_ratio to 0.7 or 0.8")
    print("3. Use this dataset with examples/train_with_dataset.py for training")
    print("4. Re-analyze periodically to ensure balanced training")
    
    return dataset, train_report


def generate_comparison():
    """Generate and compare standard vs diverse datasets."""
    
    print("\n" + "="*80)
    print("COMPARISON: Standard vs Diverse Datasets")
    print("="*80)
    
    generator = PromptDatasetGenerator(seed=42)
    
    # Generate standard dataset
    print("\n1️⃣ Generating STANDARD dataset (100 examples)...")
    standard_dataset = generator.generate_dataset(
        num_prompts=100,
        output_file="datasets/comparison/standard.json"
    )
    
    # Generate diverse dataset
    print("\n2️⃣ Generating DIVERSE dataset (100 examples)...")
    diverse_dataset = generator.generate_balanced_diverse_dataset(
        num_prompts=100,
        diverse_scenario_ratio=0.7,  # 70% diverse
        output_file="datasets/comparison/diverse.json"
    )
    
    # Analyze both
    print("\n" + "="*80)
    print("📊 STANDARD DATASET ANALYSIS")
    print("="*80)
    standard_report = analyze_dataset_diversity("datasets/comparison/standard.json")
    
    print("\n" + "="*80)
    print("📊 DIVERSE DATASET ANALYSIS")
    print("="*80)
    diverse_report = analyze_dataset_diversity("datasets/comparison/diverse.json")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    std_cov = standard_report['command_coverage']['percentage']
    div_cov = diverse_report['command_coverage']['percentage']
    
    print(f"\nCommand Coverage:")
    print(f"  Standard: {std_cov:.1f}%")
    print(f"  Diverse:  {div_cov:.1f}%")
    print(f"  Improvement: +{div_cov - std_cov:.1f}%")
    
    std_cmds = standard_report['command_coverage']['used_commands']
    div_cmds = diverse_report['command_coverage']['used_commands']
    
    print(f"\nUnique Commands:")
    print(f"  Standard: {std_cmds}")
    print(f"  Diverse:  {div_cmds}")
    print(f"  Additional: +{div_cmds - std_cmds} commands")
    
    print(f"\n✅ Diverse dataset provides {div_cov - std_cov:.1f}% better command coverage!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate diverse training dataset")
    parser.add_argument(
        '--mode',
        choices=['generate', 'compare'],
        default='generate',
        help='Mode: generate full dataset or compare standard vs diverse'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        main()
    else:
        generate_comparison()

