"""Generate new dataset with ALL scenario types."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from cli_rl_env.prompt_dataset_generator import PromptDatasetGenerator

print("=" * 80)
print("GENERATING NEW DATASET WITH ALL SCENARIO TYPES")
print("=" * 80)

# Initialize generator
gen = PromptDatasetGenerator(seed=42)

# Show available scenarios
print("\n📦 Available Scenarios:")
scenarios = gen.get_all_scenario_types()
total = 0
for gen_name, types in scenarios.items():
    count = len(types)
    total += count
    print(f"  • {gen_name}: {count} types")
print(f"  TOTAL: {total} scenario types\n")

# Generate dataset with ALL three generators
print("Generating 150 scenarios (50 each for train/val/test)...")
print("  Mix: 50% diverse (CLI), 25% Python, 25% JavaScript\n")

dataset = gen.generate_dataset(
    num_prompts=150,
    difficulty_distribution={
        'easy': 0.1,
        'medium': 0.2,
        'hard': 0.4,
        'very_hard': 0.3
    },
    generator_mix={
        'python': 0.25,
        'javascript': 0.25,
        'diverse': 0.5  # 50% CLI-focused scenarios
    },
    output_file='datasets/unified_dataset/full_dataset.json'
)

print(f"\n✅ Generated {len(dataset)} total scenarios")

# Split into train/val/test
print("\nSplitting into train/val/test...")
gen.save_dataset_splits(
    dataset,
    output_dir='datasets/unified_dataset',
    train_ratio=0.6,   # 90 examples
    val_ratio=0.2,     # 30 examples  
    test_ratio=0.2     # 30 examples
)

# Analyze scenario distribution
print("\n" + "=" * 80)
print("DATASET ANALYSIS")
print("=" * 80)

scenario_types = {}
for item in dataset:
    stype = item['metadata'].get('scenario_type', 'unknown')
    scenario_types[stype] = scenario_types.get(stype, 0) + 1

print(f"\nScenario Type Distribution ({len(scenario_types)} unique types):")
for stype, count in sorted(scenario_types.items(), key=lambda x: -x[1]):
    percentage = (count / len(dataset)) * 100
    print(f"  • {stype:25s}: {count:3d} ({percentage:5.1f}%)")

languages = {}
for item in dataset:
    lang = item['language']
    languages[lang] = languages.get(lang, 0) + 1

print(f"\nLanguage Distribution:")
for lang, count in languages.items():
    percentage = (count / len(dataset)) * 100
    print(f"  • {lang}: {count} ({percentage:.1f}%)")

difficulties = {}
for item in dataset:
    diff = item['difficulty']
    difficulties[diff] = difficulties.get(diff, 0) + 1

print(f"\nDifficulty Distribution:")
for diff in ['easy', 'medium', 'hard', 'very_hard']:
    count = difficulties.get(diff, 0)
    percentage = (count / len(dataset)) * 100 if count > 0 else 0
    print(f"  • {diff:10s}: {count} ({percentage:.1f}%)")

print("\n" + "=" * 80)
print("✅ DATASET READY FOR EVALUATION")
print("=" * 80)
print("\nDataset location: datasets/unified_dataset/")
print("  • full_dataset.json: All 150 scenarios")
print("  • train.json: 90 scenarios")
print("  • val.json: 30 scenarios")
print("  • test.json: 30 scenarios")
print("  • stats.json: Statistics")
