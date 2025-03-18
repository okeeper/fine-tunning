#!/bin/bash

echo "=================================================="
echo "  Llama模型目录检查工具"
echo "=================================================="
echo ""

# 设置默认路径
MODEL_PATH="/opt/llama/Llama-2-7b-chat"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --help)
      echo "用法: bash check_model.sh [选项]"
      echo ""
      echo "选项:"
      echo "  --path PATH    指定要检查的模型目录路径 (默认: /opt/llama/Llama-2-7b-chat)"
      echo "  --help         显示此帮助信息"
      echo ""
      echo "示例:"
      echo "  bash check_model.sh"
      echo "  bash check_model.sh --path /path/to/your/model"
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      echo "使用 --help 查看帮助信息"
      exit 1
      ;;
  esac
done

echo "检查模型目录: $MODEL_PATH"
echo ""

# 检查目录是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 目录不存在"
    echo "请确认路径是否正确，或者创建此目录并下载模型文件"
    exit 1
fi

echo "1. 基本目录检查:"
echo "✅ 目录存在"

# 检查必要的配置文件
echo ""
echo "2. 配置文件检查:"

CONFIG_FILES=("config.json" "tokenizer_config.json" "tokenizer.model" "special_tokens_map.json")
CONFIG_MISSING=false

for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$MODEL_PATH/$file" ]; then
        echo "✅ 找到: $file"
    else
        echo "❌ 缺少: $file"
        CONFIG_MISSING=true
    fi
done

# 检查模型权重文件
echo ""
echo "3. 模型权重文件检查:"

# 检查单一权重文件
WEIGHT_FILES=("pytorch_model.bin" "model.safetensors" "tf_model.h5" "model.ckpt.index" "flax_model.msgpack")
SINGLE_WEIGHT_FOUND=false

for file in "${WEIGHT_FILES[@]}"; do
    if [ -f "$MODEL_PATH/$file" ]; then
        echo "✅ 找到单一权重文件: $file"
        SINGLE_WEIGHT_FOUND=true
        break
    fi
done

# 检查分片权重文件
if [ "$SINGLE_WEIGHT_FOUND" = false ]; then
    PYTORCH_SHARDS=$(ls "$MODEL_PATH"/pytorch_model-*.bin 2>/dev/null | wc -l)
    SAFETENSORS_SHARDS=$(ls "$MODEL_PATH"/model-*.safetensors 2>/dev/null | wc -l)
    
    if [ "$PYTORCH_SHARDS" -gt 0 ]; then
        echo "✅ 找到PyTorch分片权重文件: $PYTORCH_SHARDS 个文件"
        ls "$MODEL_PATH"/pytorch_model-*.bin 2>/dev/null | while read file; do
            basename=$(basename "$file")
            echo "  - $basename"
        done
        SINGLE_WEIGHT_FOUND=true
    elif [ "$SAFETENSORS_SHARDS" -gt 0 ]; then
        echo "✅ 找到Safetensors分片权重文件: $SAFETENSORS_SHARDS 个文件"
        ls "$MODEL_PATH"/model-*.safetensors 2>/dev/null | while read file; do
            basename=$(basename "$file")
            echo "  - $basename"
        done
        SINGLE_WEIGHT_FOUND=true
    else
        echo "❌ 未找到任何模型权重文件"
    fi
fi

# 汇总状态
echo ""
echo "4. 诊断汇总:"

if [ "$CONFIG_MISSING" = true ]; then
    echo "⚠️  缺少部分配置文件 - 可能导致问题"
else
    echo "✅ 所有配置文件都存在"
fi

if [ "$SINGLE_WEIGHT_FOUND" = false ]; then
    echo "❌ 未找到任何模型权重文件 - 这将导致加载失败"
    echo ""
    echo "解决方案:"
    echo "1. 确保您已经下载了完整的模型文件"
    echo "2. 如果已下载，请检查文件名是否符合预期"
    echo "3. 可以运行 'fix_model_path.sh --model_path \"$MODEL_PATH\"' 修复配置"
    echo "4. 或者使用Hugging Face模型ID代替本地路径"
else
    echo "✅ 找到模型权重文件 - 应该可以加载"
fi

# 检查目录权限
echo ""
echo "5. 权限检查:"
if [ -r "$MODEL_PATH" ]; then
    echo "✅ 目录具有读取权限"
else
    echo "❌ 目录缺少读取权限"
fi

# 如果发现重大问题，提供进一步建议
if [ "$SINGLE_WEIGHT_FOUND" = false ] || [ "$CONFIG_MISSING" = true ]; then
    echo ""
    echo "=================================================="
    echo "  建议操作"
    echo "=================================================="
    echo ""
    echo "您的模型目录似乎有问题。建议采取以下措施之一:"
    echo ""
    echo "1. 修复本地模型目录:"
    echo "   - 确保下载了完整的模型文件"
    echo "   - 确保文件名称正确"
    echo "   - 运行 'fix_model_path.sh --model_path \"$MODEL_PATH\"' 更新配置"
    echo ""
    echo "2. 使用Hugging Face模型ID替代本地路径:"
    echo "   - 运行 'fix_model_path.sh' 将自动切换到使用Hugging Face模型ID"
    echo "   - 或者修改配置文件中的 model_name_or_path 为 'meta-llama/Llama-2-7b-chat-hf'"
else
    echo ""
    echo "=================================================="
    echo "  模型目录检查完成"
    echo "=================================================="
    echo ""
    echo "您的模型目录看起来配置正确。如果仍然遇到问题，可以:"
    echo "1. 检查更多细节，如模型版本是否匹配预期"
    echo "2. 确认您有足够的权限访问这些文件"
    echo "3. 您可以运行脚本验证此目录: 'fix_model_path.sh --model_path \"$MODEL_PATH\"'"
fi 