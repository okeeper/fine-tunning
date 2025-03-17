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
        python data/prepare_dataset.py --config_file "../$CONFIG_FILE" --output_dir "$DATA_DIR"
        
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
echo "运行命令: python src/finetune.py --config_file $CONFIG_FILE --data_dir $DATA_DIR $EXTRA_ARGS"
echo ""

python src/finetune.py --config_file "$CONFIG_FILE" --data_dir "$DATA_DIR" $EXTRA_ARGS

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