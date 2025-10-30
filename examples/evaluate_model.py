"""Example script for evaluating models on CLI tasks using vLLM."""

import argparse
from pathlib import Path

from cli_rl_env.evaluation import ModelEvaluator, EvaluationMetrics
from cli_rl_env.evaluation.evaluator import ModelConfig


def main():
    """Run model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate model on CLI tasks using vLLM"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'openai/gpt-4' or 'hosted-vllm/Qwen/Qwen2.5-Coder-32B-Instruct')"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL for vLLM (e.g., 'http://localhost:8000/v1')"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key (default: 'EMPTY' for vLLM)"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/diverse_dataset/test.json",
        help="Path to test dataset"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all)"
    )
    parser.add_argument(
        "--max-commands",
        type=int,
        default=50,
        help="Maximum commands per episode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results/results.json",
        help="Output path for results"
    )
    
    # Model parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens per response"
    )
    
    args = parser.parse_args()
    
    # Create model config
    model_config = ModelConfig(
        model_name=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_config=model_config,
        max_commands=args.max_commands,
        verbose=True
    )
    
    # Run evaluation
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80)
    print(f"Model: {args.model}")
    if args.api_base:
        print(f"vLLM Endpoint: {args.api_base}")
    print(f"Dataset: {args.dataset}")
    print(f"Max Commands: {args.max_commands}")
    print("="*80 + "\n")
    
    try:
        results = evaluator.evaluate_dataset(
            dataset_path=args.dataset,
            num_examples=args.num_examples,
            output_path=args.output
        )
        
        # Print detailed report
        print("\n" + evaluator.metrics.get_detailed_report())
        
        print(f"\n✅ Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

