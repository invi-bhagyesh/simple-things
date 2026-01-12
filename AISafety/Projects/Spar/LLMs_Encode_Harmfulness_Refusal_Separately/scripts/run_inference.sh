#!/bin/bash
# Run Inference Script
# Usage: ./scripts/run_inference.sh MODEL_PATH INPUT_FILE OUTPUT_FILE [MODEL_TYPE]
#
# Example:
#   ./scripts/run_inference.sh NousResearch/Llama-2-7b-chat-hf data/xstest-harmless.json outputs/xstest_results.jsonl llama2

MODEL_PTH=${1:-"NousResearch/Llama-2-7b-chat-hf"}
INPUT_PTH=${2:-"data/advbench.json"}  
OUTPUT_PTH=${3:-"outputs/inference_results.jsonl"}
MODEL_TYPE=${4:-"llama2"}

echo "Running inference..."
echo "  Model: $MODEL_PTH"
echo "  Input: $INPUT_PTH"
echo "  Output: $OUTPUT_PTH"
echo "  Model Type: $MODEL_TYPE"

python scripts/run_inference.py \
    --model "$MODEL_PTH" \
    --input "$INPUT_PTH" \
    --output "$OUTPUT_PTH" \
    --model_type "$MODEL_TYPE" \
    --max_new_tokens 256 \
    --batch_size 1

echo "Inference complete!"
