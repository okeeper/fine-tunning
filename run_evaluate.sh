#!/bin/bash

# 运行评估脚本

BASE_MODEL_PATH="/opt/llama/Llama-2-7b-chat-hf"
FINETUNED_MODEL_PATH="./output/llama2-7b-chat-lccc"
RESULT_DIR="./evaluation_results"

# 微调数据测试集和标准测试集路径
FINETUNE_TEST_FILE="./data/processed/test.json"  # 微调数据的测试集
STANDARD_TEST_FILE="./data/standard_benchmarks/test.json"  # 用于评估灾难性遗忘的标准测试集
PREPARED_TEST_DIR="./data/processed/test"  # prepare_dataset.py生成的测试数据目录

# 运行评估
echo "开始评估..."
python src/evaluate.py \
    --base_model_path $BASE_MODEL_PATH \
    --finetuned_model_path $FINETUNED_MODEL_PATH \
    --finetune_test_file $FINETUNE_TEST_FILE \
    --standard_test_file $STANDARD_TEST_FILE \
    --prepared_test_dir $PREPARED_TEST_DIR \
    --result_dir $RESULT_DIR

echo "评估完成，结果保存在 $RESULT_DIR 目录" 