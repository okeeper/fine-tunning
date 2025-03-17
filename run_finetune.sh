#!/bin/bash

# Llama-2-7b-chat模型微调训练脚本
# 这个脚本会自动完成以下步骤：
# 1. 准备数据集（如果需要）
# 2. 运行微调训练
# 3. 输出训练结果和后续步骤提示

# 设置默认参数
CONFIG_FILE="config/finetune_config.json"
DATA_DIR="./data/processed"
PREPARE_DATA=true
USE_CPU=false
MAX_SAMPLES=10000

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --config_file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --skip_data_prep)
      PREPARE_DATA=false
      shift
      ;;
    --use_cpu)
      USE_CPU=true
      shift
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --help)
      echo "用法: bash run_finetune.sh [选项]"
      echo ""
      echo "选项:"
      echo "  --config_file FILE    指定配置文件路径 (默认: config/finetune_config.json)"
      echo "  --data_dir DIR        指定数据目录 (默认: ./data/processed)"
      echo "  --skip_data_prep      跳过数据准备步骤"
      echo "  --use_cpu             使用CPU进行训练 (不使用量化)"
      echo "  --max_samples N       限制训练样本数量 (默认: 10000)"
      echo "  --help                显示此帮助信息"
      echo ""
      echo "示例:"
      echo "  bash run_finetune.sh --use_cpu --max_samples 1000"
      echo "  bash run_finetune.sh --skip_data_prep --learning_rate 1e-5"
      exit 0
      ;;
    *)
      # 将其他参数保存起来，传递给微调脚本
      EXTRA_ARGS="$EXTRA_ARGS $1"
      shift
      ;;
  esac
done

# 显示欢迎信息
echo "=================================================="
echo "  Llama-2-7b-chat 中文对话模型微调训练"
echo "=================================================="
echo ""

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    echo "请确保配置文件路径正确，或使用 --config_file 参数指定配置文件"
    exit 1
fi

echo "使用配置文件: $CONFIG_FILE"
echo "数据目录: $DATA_DIR"
if [ "$USE_CPU" = true ]; then
    echo "训练模式: CPU (不使用量化)"
    echo "警告: CPU模式下训练速度会非常慢，建议减小样本数量"
else
    echo "训练模式: GPU (使用4bit量化)"
fi
echo ""

# 准备数据集（如果需要）
if [ "$PREPARE_DATA" = true ]; then
    echo "步骤1: 准备数据集..."
    
    # 检查数据目录是否已存在
    if [ -d "$DATA_DIR/train" ] && [ -d "$DATA_DIR/validation" ]; then
        echo "数据集已存在于 $DATA_DIR"
        echo "如果需要重新准备数据集，请删除该目录或指定新的数据目录"
        echo ""
    else
        echo "开始准备数据集..."
        DATASET_ARGS="--config_file ../$CONFIG_FILE --output_dir $DATA_DIR"
        
        # 如果指定了最大样本数，添加到参数中
        if [ "$MAX_SAMPLES" != "10000" ]; then
            DATASET_ARGS="$DATASET_ARGS --max_samples $MAX_SAMPLES"
        fi
        
        python data/prepare_dataset.py $DATASET_ARGS
        
        # 检查数据准备是否成功
        if [ $? -ne 0 ]; then
            echo "错误: 数据准备失败"
            exit 1
        fi
        
        echo "数据集准备完成！"
        echo ""
    fi
else
    echo "跳过数据准备步骤"
    echo ""
fi

# 运行微调训练
echo "步骤2: 开始微调训练..."

# 构建微调命令
FINETUNE_CMD="python src/finetune.py --config_file $CONFIG_FILE --data_dir $DATA_DIR"

# 如果使用CPU模式，添加参数
if [ "$USE_CPU" = true ]; then
    FINETUNE_CMD="$FINETUNE_CMD --use_cpu"
fi

# 添加额外参数
FINETUNE_CMD="$FINETUNE_CMD $EXTRA_ARGS"

echo "运行命令: $FINETUNE_CMD"
echo ""

# 执行微调命令
$FINETUNE_CMD

# 检查训练是否成功
if [ $? -ne 0 ]; then
    echo "错误: 微调训练失败"
    exit 1
fi

echo ""
echo "=================================================="
echo "  微调训练完成！"
echo "=================================================="
echo ""
echo "您可以使用以下命令评估模型性能:"
echo "python src/evaluate.py --model_path [模型输出目录]"
echo ""
echo "如需在新数据上测试模型，可以使用:"
echo "python src/generate.py --model_path [模型输出目录] --prompt \"您的提示\""
echo "" 