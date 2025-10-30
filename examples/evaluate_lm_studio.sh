#!/bin/bash
# Quick evaluation script for LM Studio models

# Configuration
MODEL=${1:-"openai/gpt-oss-20b"}
API_BASE=${2:-"http://localhost:1234/v1"}
NUM_EXAMPLES=${3:-10}

echo "========================================"
echo "LM Studio Model Evaluation"
echo "========================================"
echo "Model: $MODEL"
echo "API Base: $API_BASE"
echo "Examples: $NUM_EXAMPLES"
echo "========================================"
echo ""

# Check if LM Studio is running
echo "Checking LM Studio server..."
if curl -s "${API_BASE}/models" > /dev/null; then
    echo "✅ LM Studio server is responding"
else
    echo "❌ Cannot connect to LM Studio server at $API_BASE"
    echo ""
    echo "   Please start LM Studio first:"
    echo "   1. Open LM Studio application"
    echo "   2. Load a model (e.g., Qwen/Qwen2.5-Coder)"
    echo "   3. Go to 'Local Server' tab"
    echo "   4. Click 'Start Server' (port 1234)"
    exit 1
fi

echo ""

# Run evaluation
python examples/evaluate_model.py \
    --model "$MODEL" \
    --api-base "$API_BASE" \
    --api-key "lm-studio" \
    --num-examples "$NUM_EXAMPLES" \
    --max-commands 50 \
    --temperature 0.3 \
    --output "evaluation_results/lm_studio_$(date +%Y%m%d_%H%M%S).json"


# Example evaluation
# uv run python examples/evaluate_model.py \
#     --model "openai/gpt-oss-20b" \
#     --api-base "http://localhost:1234/v1" \
#     --api-key "lm-studio" \
#     --dataset "datasets/diverse_dataset/test.json" \
#     --num-examples 10 \
#     --temperature 0.3 \
#     --max-commands 50 \
#     --output "evaluation_results/lm_studio_$(date +%Y%m%d_%H%M%S).json"