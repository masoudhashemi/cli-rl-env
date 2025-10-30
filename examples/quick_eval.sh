#!/bin/bash
# Quick evaluation script for vLLM models

# Configuration
MODEL=${1:-"Qwen/Qwen2.5-Coder-32B-Instruct"}
API_BASE=${2:-"http://localhost:8000/v1"}
NUM_EXAMPLES=${3:-10}

echo "========================================"
echo "Quick Model Evaluation"
echo "========================================"
echo "Model: $MODEL"
echo "API Base: $API_BASE"
echo "Examples: $NUM_EXAMPLES"
echo "========================================"
echo ""

# Check if vLLM is running
echo "Checking vLLM server..."
if curl -s "${API_BASE}/models" > /dev/null; then
    echo "✅ vLLM server is responding"
else
    echo "❌ Cannot connect to vLLM server at $API_BASE"
    echo "   Please start vLLM first:"
    echo "   python -m vllm.entrypoints.openai.api_server --model $MODEL --port 8000"
    exit 1
fi

echo ""

# Run evaluation
python examples/evaluate_model.py \
    --model "hosted-vllm/$MODEL" \
    --api-base "$API_BASE" \
    --num-examples "$NUM_EXAMPLES" \
    --max-commands 50 \
    --temperature 0.3 \
    --output "evaluation_results/quick_eval_$(date +%Y%m%d_%H%M%S).json"

