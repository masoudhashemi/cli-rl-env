"""Demonstration of the evaluation module capabilities."""

import json
from cli_rl_env.evaluation import ModelEvaluator
from cli_rl_env.evaluation.evaluator import ModelConfig


def main():
    """Demo evaluation workflow."""
    
    print("\n" + "="*80)
    print("EVALUATION MODULE DEMONSTRATION")
    print("="*80 + "\n")
    
    # Example 1: Show configuration
    print("📋 STEP 1: Configure Model Connection")
    print("-" * 80)
    
    print("\n🔧 EXAMPLE A: LM Studio Configuration")
    print("For LM Studio running on http://localhost:1234:")
    print()
    
    lm_studio_config = ModelConfig(
        model_name="openai/gpt-oss-20b",  # Or whatever model you loaded in LM Studio
        api_base="http://localhost:1234/v1",
        api_key="lm-studio",  # LM Studio accepts any key
        temperature=0.3,
        max_tokens=2048,
        timeout=120
    )
    
    print(f"  Model: {lm_studio_config.model_name}")
    print(f"  API Base: {lm_studio_config.api_base}")
    print(f"  API Key: {lm_studio_config.api_key}")
    print(f"  Temperature: {lm_studio_config.temperature}")
    print(f"  Max Tokens: {lm_studio_config.max_tokens}")
    
    print("\n🔧 EXAMPLE B: vLLM Configuration")
    print("For vLLM running on http://localhost:8000:")
    print()
    
    vllm_config = ModelConfig(
        model_name="hosted-vllm/Qwen/Qwen2.5-Coder-32B-Instruct",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
        temperature=0.3,
        max_tokens=2048,
        timeout=120
    )
    
    print(f"  Model: {vllm_config.model_name}")
    print(f"  API Base: {vllm_config.api_base}")
    print(f"  Temperature: {vllm_config.temperature}")
    
    # Example 2: Show evaluator setup
    print("\n📊 STEP 2: Create Evaluator")
    print("-" * 80)
    
    print("Code example:")
    print("""
    evaluator = ModelEvaluator(
        model_config=config,
        max_commands=50,
        verbose=True
    )
    """)
    
    print("Evaluator configured with:")
    print(f"  - Max commands per task: 50")
    print(f"  - Verbose output: Enabled")
    print(f"  - Metrics tracking: Enabled")
    
    print("\nNote: Install litellm to run actual evaluation: pip install litellm")
    
    # Example 3: Show single scenario evaluation (mock)
    print("\n🔍 STEP 3: How a Single Scenario is Evaluated")
    print("-" * 80)
    
    print("""
Evaluation Flow for Each Scenario:
    
1. Environment Setup
   └─> Create sandbox with initial files
   
2. Send Task to Model
   ├─> System prompt (you are a CLI expert...)
   ├─> Task description
   └─> Available files
   
3. Model Response Loop:
   ├─> Model returns command
   ├─> Execute command in sandbox
   ├─> Capture output/errors
   ├─> Send feedback to model
   └─> Repeat until done or max commands
   
4. Verification
   ├─> Run verification rules (tests, text matching, etc.)
   ├─> Calculate success/failure
   └─> Record metrics
   
5. Save Results
   └─> Command history, verification, timing, etc.
""")
    
    # Example 4: Show what metrics are tracked
    print("\n📈 STEP 4: Metrics Tracked")
    print("-" * 80)
    
    print("""
Overall:
  • Total examples evaluated
  • Pass rate (%)
  • Average commands used
  • Average time per task
  
By Difficulty:
  • Easy, Medium, Hard, Very Hard
  • Pass rate for each
  
By Scenario Type:
  • Bug-fixing (calculator, data_processor, etc.)
  • File operations (comparison, archive, etc.)
  • Text processing (grep, sed, awk, etc.)
  • Git workflows
  • Pass rate for each type
  
Per Example:
  • Commands executed
  • Verification results
  • Time taken
  • Success/failure
""")
    
    # Example 5: Show sample output
    print("\n💾 STEP 5: Sample Output")
    print("-" * 80)
    
    sample_result = {
        "scenario_id": "prompt_000123",
        "success": True,
        "num_commands": 7,
        "commands": [
            "ls -la",
            "cat calculator.py",
            "grep 'def' calculator.py",
            "sed -i 's/a - b/a + b/g' calculator.py",
            "pytest test_calculator.py",
            "cat calculator.py",
            "DONE"
        ],
        "verification": {
            "test_results": {
                "success": True,
                "passed": 5,
                "failed": 0,
                "total": 5
            }
        },
        "time_seconds": 12.5,
        "difficulty": "medium",
        "scenario_type": "calculator"
    }
    
    print("Example result for one scenario:")
    print(json.dumps(sample_result, indent=2))
    
    # Example 6: Show usage
    print("\n\n🚀 STEP 6: Run Actual Evaluation")
    print("-" * 80)
    
    print("\n" + "="*80)
    print("OPTION A: Evaluate with LM Studio")
    print("="*80)
    print("""
1. Start LM Studio:
   - Open LM Studio application
   - Load your model (e.g., Qwen/Qwen2.5-Coder or any coding model)
   - Go to "Local Server" tab
   - Click "Start Server" (default port 1234)

2. Verify LM Studio is running:
   curl http://localhost:1234/v1/models

3. Run evaluation:
   python examples/evaluate_model.py \\
       --model "openai/gpt-oss-20b" \\
       --api-base "http://localhost:1234/v1" \\
       --api-key "lm-studio" \\
       --dataset "datasets/diverse_dataset/test.json" \\
       --num-examples 10 \\
       --temperature 0.3 \\
       --output "evaluation_results/lm_studio_results.json"

   Note: The model name "openai/gpt-oss-20b" should match what's shown in LM Studio

4. Check results:
   cat evaluation_results/lm_studio_results.json
""")
    
    print("\n" + "="*80)
    print("OPTION B: Evaluate with vLLM")
    print("="*80)
    print("""
1. Start vLLM:
   python -m vllm.entrypoints.openai.api_server \\
       --model Qwen/Qwen2.5-Coder-32B-Instruct \\
       --port 8000

2. Quick test (10 examples):
   ./examples/quick_eval.sh

3. Full evaluation:
   python examples/evaluate_model.py \\
       --model "hosted-vllm/Qwen/Qwen2.5-Coder-32B-Instruct" \\
       --api-base "http://localhost:8000/v1" \\
       --dataset "datasets/diverse_dataset/test.json" \\
       --num-examples 10 \\
       --output "evaluation_results/vllm_results.json"

4. Check results:
   cat evaluation_results/vllm_results.json
""")
    
    print("\n" + "="*80)
    print("See EVALUATION_GUIDE.md for complete documentation")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

