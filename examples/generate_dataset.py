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
    parser = argparse.ArgumentParser(description="Generate fresh CLI RL dataset")
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

    args = parser.parse_args()
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

        scenarios.append(scenario_to_dict(s, sid))

    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"Generated {len(scenarios)} scenarios -> {out_path}")


if __name__ == "__main__":
    main()
