#!/bin/bash

# 运行评估脚本

MODEL_PATH="./output/llama2-7b-chat-lccc"
BASE_MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"

# 运行评估
echo "开始评估..."
python src/evaluate.py \
    --model_path $MODEL_PATH \
    --base_model_path $BASE_MODEL_PATH \
    --eval_data_dir ./data/processed \
    --output_dir ./evaluation_results \
    --eval_original_tasks \
    --perplexity_eval \
    --knowledge_eval \
    --overfitting_analysis

echo "评估完成，结果保存在 ./evaluation_results 目录" 