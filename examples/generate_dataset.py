"""Generate a fresh dataset of CLI RL scenarios.

This script uses existing scenario generators to create a JSON dataset
compatible with examples/evaluate_model.py.
"""

import argparse
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any

from cli_rl_env.scenario_generator.base import DifficultyLevel, Scenario, FileContent, VerificationRule
from cli_rl_env.scenario_generator.python_generator import PythonScenarioGenerator
from cli_rl_env.scenario_generator.javascript_generator import JavaScriptScenarioGenerator
from cli_rl_env.scenario_generator.diverse_scenarios import DiverseScenarioGenerator


def scenario_to_dict(s: Scenario, id_str: str) -> Dict[str, Any]:
    return {
        "id": id_str,
        "difficulty": s.difficulty.value,
        "language": s.language,
        "task_description": s.task_description,
        "files": [
            {"path": f.path, "content": f.content, "is_test": bool(f.is_test)}
            for f in s.files
        ],
        "cli_history": list(s.cli_history or []),
        "expected_commands": int(s.expected_commands),
        "verification_rules": [
            {
                "type": vr.type,
                "target": vr.target,
                "expected": vr.expected,
                "description": vr.description,
            }
            for vr in s.verification_rules
        ],
        "metadata": s.metadata or {},
    }


def main():
    parser = argparse.ArgumentParser(description="Generate fresh CLI RL dataset using ALL scenario generators")
    parser.add_argument("--count", type=int, default=200, help="Number of scenarios to generate")
    parser.add_argument(
        "--output", type=str,
        default=f"datasets/diverse_dataset/generated_{time.strftime('%Y%m%d_%H%M%S')}.json",
        help="Output JSON path"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--mix", type=str, default="diverse:0.5,python:0.25,javascript:0.25",
        help="Mix of generators with weights, e.g., 'diverse:0.5,python:0.25,javascript:0.25'"
    )
    parser.add_argument(
        "--list-scenarios", action="store_true",
        help="List all available scenario types and exit"
    )

    args = parser.parse_args()
    
    # List scenarios if requested
    if args.list_scenarios:
        from cli_rl_env.prompt_dataset_generator import PromptDatasetGenerator
        gen = PromptDatasetGenerator()
        scenarios = gen.get_all_scenario_types()
        
        print("=" * 80)
        print("ALL AVAILABLE SCENARIO TYPES")
        print("=" * 80)
        
        total = 0
        for gen_name, types in scenarios.items():
            print(f"\n{gen_name.upper().replace('_', ' ')} ({len(types)} types):")
            print("-" * 80)
            for t in types:
                print(f"  • {t}")
            total += len(types)
        
        print("\n" + "=" * 80)
        print(f"TOTAL: {total} unique scenario types across all generators")
        print("=" * 80)
        return
    
    if args.seed is not None:
        random.seed(args.seed)

    # Parse mix
    parts = [p.strip() for p in args.mix.split(",") if p.strip()]
    weights = {}
    for p in parts:
        name, w = p.split(":")
        weights[name.strip()] = float(w)
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("Invalid --mix weights")

    print("=" * 80)
    print("GENERATING UNIFIED DATASET FROM ALL SCENARIO GENERATORS")
    print("=" * 80)
    print(f"\nGenerator Mix:")
    for name, w in weights.items():
        percentage = (w / total_w) * 100
        print(f"  • {name}: {percentage:.1f}%")
    print(f"\nTotal scenarios to generate: {args.count}")
    print()

    # Generators
    py_gen = PythonScenarioGenerator(seed=args.seed)
    js_gen = JavaScriptScenarioGenerator(seed=args.seed)
    div_gen = DiverseScenarioGenerator(seed=args.seed)

    def pick_generator_name() -> str:
        r = random.random() * total_w
        acc = 0.0
        for name, w in weights.items():
            acc += w
            if r <= acc:
                return name
        return list(weights.keys())[-1]

    def pick_difficulty() -> DifficultyLevel:
        return random.choice([
            DifficultyLevel.EASY,
            DifficultyLevel.MEDIUM,
            DifficultyLevel.HARD,
            DifficultyLevel.VERY_HARD,
        ])

    def pick_language() -> str:
        return random.choice(["python", "javascript"])

    scenarios: List[Dict[str, Any]] = []
    ts = time.strftime('%Y%m%d_%H%M%S')
    
    # Track scenario type distribution
    scenario_type_counts = {}

    for i in range(1, args.count + 1):
        gen_name = pick_generator_name()
        difficulty = pick_difficulty()
        sid = f"gen_{ts}_{i:04d}"

        if gen_name == "python":
            s = py_gen.generate(difficulty)
        elif gen_name == "javascript":
            s = js_gen.generate(difficulty)
        else:  # diverse
            lang = pick_language()
            s = div_gen.generate_diverse_scenario(difficulty, lang)
        
        # Track scenario types
        scenario_type = s.metadata.get('scenario_type', 'unknown')
        scenario_type_counts[scenario_type] = scenario_type_counts.get(scenario_type, 0) + 1

        scenarios.append(scenario_to_dict(s, sid))
        
        if i % 50 == 0:
            print(f"  Generated {i}/{args.count} scenarios...")

    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"\n✅ Generated {len(scenarios)} scenarios -> {out_path}")
    print(f"\nScenario Type Distribution:")
    for stype, count in sorted(scenario_type_counts.items(), key=lambda x: -x[1]):
        print(f"  • {stype}: {count}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
