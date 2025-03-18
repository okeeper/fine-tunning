#!/bin/bash

echo "=================================================="
echo "  Meta格式LLaMA模型转换为Hugging Face格式"
echo "=================================================="
echo ""

# 默认参数
INPUT_DIR=""
OUTPUT_DIR=""
MODEL_SIZE="7B"
CHAT_MODEL=false
INSTALL_DEPS=true

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --input_dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model_size)
      MODEL_SIZE="$2"
      shift 2
      ;;
    --chat_model)
      CHAT_MODEL=true
      shift
      ;;
    --no_install_deps)
      INSTALL_DEPS=false
      shift
      ;;
    --help)
      echo "用法: ./convert_meta_to_hf.sh [选项]"
      echo ""
      echo "选项:"
      echo "  --input_dir DIR       指定包含Meta格式LLaMA模型的目录(必需)"
      echo "  --output_dir DIR      指定输出Hugging Face格式模型的目录(必需)"
      echo "  --model_size SIZE     指定模型大小, 如 7B, 13B, 70B (默认: 7B)"
      echo "  --chat_model          指定是否为chat模型(添加此参数表示是chat模型)"
      echo "  --no_install_deps     不自动安装依赖"
      echo "  --help                显示此帮助信息"
      echo ""
      echo "示例:"
      echo "  ./convert_meta_to_hf.sh --input_dir /opt/llama/Llama-2-7b-chat --output_dir /opt/llama/Llama-2-7b-chat-hf --chat_model"
      echo "  ./convert_meta_to_hf.sh --input_dir /opt/llama/Llama-2-13b --output_dir /opt/llama/Llama-2-13b-hf --model_size 13B"
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      echo "使用 --help 查看帮助信息"
      exit 1
      ;;
  esac
done

# 检查必需参数
if [ -z "$INPUT_DIR" ]; then
    echo "错误: 必须指定输入目录 (--input_dir)"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "错误: 必须指定输出目录 (--output_dir)"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录 $INPUT_DIR 不存在"
    exit 1
fi

# 打印设置
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "模型大小: $MODEL_SIZE"
echo "Chat模型: $([ "$CHAT_MODEL" = true ] && echo "是" || echo "否")"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 创建临时目录
TEMP_DIR=$(mktemp -d)
echo "创建临时目录: $TEMP_DIR"
echo ""

# 清理函数
cleanup() {
    echo "清理临时文件..."
    rm -rf "$TEMP_DIR"
    echo "清理完成"
}

# 设置退出时调用清理函数
trap cleanup EXIT

# 检查并安装依赖
if [ "$INSTALL_DEPS" = true ]; then
    echo "步骤1: 检查并安装依赖..."
    
    # 检查pip是否存在
    if ! command -v pip &> /dev/null; then
        echo "未找到pip，请先安装Python和pip"
        exit 1
    fi
    
    # 安装依赖
    echo "安装必要的Python库..."
    pip install transformers torch numpy sentencepiece
    
    # 安装Hugging Face转换工具
    echo "安装Hugging Face转换工具..."
    pip install huggingface_hub
    
    echo "依赖安装完成"
else
    echo "跳过依赖安装"
fi
echo ""

# 确认输入目录包含Meta格式的文件
echo "步骤2: 验证输入目录包含Meta格式的LLaMA模型..."

CONSOLIDATED_FILE="$INPUT_DIR/consolidated.00.pth"
PARAMS_FILE="$INPUT_DIR/params.json"
TOKENIZER_FILE="$INPUT_DIR/tokenizer.model"

if [ ! -f "$CONSOLIDATED_FILE" ]; then
    echo "错误: 未找到模型权重文件 (consolidated.00.pth)"
    echo "请确认输入路径包含Meta格式的LLaMA模型"
    exit 1
fi

if [ ! -f "$PARAMS_FILE" ]; then
    echo "错误: 未找到参数文件 (params.json)"
    exit 1
fi

if [ ! -f "$TOKENIZER_FILE" ]; then
    echo "错误: 未找到分词器文件 (tokenizer.model)"
    exit 1
fi

echo "✓ 输入目录包含有效的Meta格式LLaMA模型"
echo ""

# 创建转换脚本
echo "步骤3: 创建Python转换脚本..."

cat > "$TEMP_DIR/convert.py" << 'EOF'
import os
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from sentencepiece import SentencePieceProcessor

def load_meta_model(model_path: str, model_size: str) -> Tuple[dict, dict]:
    """加载Meta格式的模型权重和参数"""
    print(f"加载Meta格式的LLaMA模型 ({model_size})...")
    
    # 加载参数
    params_path = os.path.join(model_path, "params.json")
    with open(params_path, "r") as f:
        params = json.load(f)
    
    # 加载权重
    consolidated_path = os.path.join(model_path, "consolidated.00.pth")
    checkpoints = torch.load(consolidated_path, map_location="cpu")
    
    return checkpoints, params

def get_model_config(params: dict, model_size: str, chat_model: bool) -> LlamaConfig:
    """创建Hugging Face模型配置"""
    print("创建模型配置...")
    
    # 根据模型大小调整参数
    if model_size == "7B":
        dim = 4096
        n_layers = 32
        n_heads = 32
        intermediate_size = 11008
    elif model_size == "13B":
        dim = 5120
        n_layers = 40
        n_heads = 40
        intermediate_size = 13824
    elif model_size == "70B":
        dim = 8192
        n_layers = 80
        n_heads = 64
        intermediate_size = 28672
    else:
        # 从params.json读取
        dim = params.get("dim", 4096)
        n_layers = params.get("n_layers", 32)
        n_heads = params.get("n_heads", 32)
        intermediate_size = None  # 将根据dim计算
    
    # 计算intermediate_size如果未指定
    if intermediate_size is None:
        # 通常是dim的2.7倍左右
        intermediate_size = int(dim * 2.7)
    
    # 创建配置
    config = LlamaConfig(
        vocab_size=params.get("vocab_size", 32000),
        hidden_size=dim,
        num_attention_heads=n_heads,
        num_hidden_layers=n_layers,
        intermediate_size=intermediate_size,
        max_position_embeddings=params.get("max_seq_len", 4096),
        rms_norm_eps=params.get("norm_eps", 1e-5),
        num_key_value_heads=n_heads,  # 可能需要根据模型调整
        rope_theta=params.get("rope_freq_base", 10000),
        rope_scaling=None,  # 可选配置
        attention_bias=False,
    )
    
    # 如果是chat模型，添加相关配置
    if chat_model:
        config.eos_token_id = 2
        config.pad_token_id = 0
        config.bos_token_id = 1
        config.chat_template = "{% for message in messages %}\n{% if message['role'] == 'system' %}\n<s>[INST] <<SYS>>\n{{ message['content'] }}\n<</SYS>>\n\n{% elif message['role'] == 'user' %}\n{% if loop.index > 1 %}</s>{% endif %}\n<s>[INST] {{ message['content'] }} [/INST]\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] }}\n{% endif %}\n{% endfor %}"
    
    return config

def convert_weights(meta_weights: dict, config: LlamaConfig) -> dict:
    """转换Meta格式的权重到Hugging Face格式"""
    print("转换模型权重...")
    
    # 创建空的HF模型
    hf_model = LlamaForCausalLM(config)
    hf_state_dict = hf_model.state_dict()
    
    # 权重映射关系 (Meta格式 -> HF格式)
    weight_map = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    
    # 层内参数映射
    for layer_idx in range(config.num_hidden_layers):
        # 注意力相关参数
        weight_map.update({
            f"layers.{layer_idx}.attention.wq.weight": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"layers.{layer_idx}.attention.wk.weight": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"layers.{layer_idx}.attention.wv.weight": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            f"layers.{layer_idx}.attention.wo.weight": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
        })
        
        # FFN参数
        weight_map.update({
            f"layers.{layer_idx}.feed_forward.w1.weight": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
            f"layers.{layer_idx}.feed_forward.w2.weight": f"model.layers.{layer_idx}.mlp.down_proj.weight",
            f"layers.{layer_idx}.feed_forward.w3.weight": f"model.layers.{layer_idx}.mlp.up_proj.weight",
        })
        
        # 层归一化
        weight_map.update({
            f"layers.{layer_idx}.attention_norm.weight": f"model.layers.{layer_idx}.input_layernorm.weight",
            f"layers.{layer_idx}.ffn_norm.weight": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        })
    
    # 执行权重转换
    new_state_dict = {}
    for meta_name, meta_tensor in meta_weights.items():
        if meta_name in weight_map:
            hf_name = weight_map[meta_name]
            if meta_tensor.shape != hf_state_dict[hf_name].shape:
                print(f"警告: 维度不匹配 {meta_name} -> {hf_name}: {meta_tensor.shape} vs {hf_state_dict[hf_name].shape}")
                # 可能需要对某些维度进行转置或重塑
                if len(meta_tensor.shape) == 2 and len(hf_state_dict[hf_name].shape) == 2:
                    if meta_tensor.shape[0] == hf_state_dict[hf_name].shape[1] and meta_tensor.shape[1] == hf_state_dict[hf_name].shape[0]:
                        print(f"  执行转置 {meta_tensor.shape} -> {hf_state_dict[hf_name].shape}")
                        meta_tensor = meta_tensor.T
            
            new_state_dict[hf_name] = meta_tensor
        else:
            print(f"跳过未映射的参数: {meta_name}")
    
    return new_state_dict

def prepare_tokenizer(model_path: str, output_dir: str, chat_model: bool):
    """准备分词器文件"""
    print("准备分词器...")
    
    # 复制分词器文件
    os.system(f"cp {model_path}/tokenizer.model {output_dir}/")
    
    # 创建tokenizer_config.json
    tok_config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": {
            "__type": "AddedToken",
            "content": "<s>",
            "lstrip": False,
            "normalized": True,
            "rstrip": False,
            "single_word": False
        },
        "clean_up_tokenization_spaces": False,
        "eos_token": {
            "__type": "AddedToken",
            "content": "</s>",
            "lstrip": False,
            "normalized": True,
            "rstrip": False,
            "single_word": False
        },
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "padding_side": "right",
        "special_tokens_map_file": "special_tokens_map.json",
        "tokenizer_class": "LlamaTokenizer",
        "unk_token": {
            "__type": "AddedToken",
            "content": "<unk>",
            "lstrip": False,
            "normalized": True,
            "rstrip": False,
            "single_word": False
        }
    }
    
    # 为chat模型添加特殊配置
    if chat_model:
        tok_config["chat_template"] = "{% for message in messages %}\n{% if message['role'] == 'system' %}\n<s>[INST] <<SYS>>\n{{ message['content'] }}\n<</SYS>>\n\n{% elif message['role'] == 'user' %}\n{% if loop.index > 1 %}</s>{% endif %}\n<s>[INST] {{ message['content'] }} [/INST]\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] }}\n{% endif %}\n{% endfor %}"
    
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tok_config, f, indent=2)
    
    # 创建special_tokens_map.json
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>"
    }
    
    with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens, f, indent=2)
    
    # 尝试加载分词器测试
    try:
        tokenizer = LlamaTokenizer.from_pretrained(output_dir)
        print(f"✓ 分词器加载测试成功")
    except Exception as e:
        print(f"× 分词器加载测试失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="将Meta格式的LLaMA模型转换为Hugging Face格式")
    parser.add_argument("--input_dir", type=str, required=True, help="Meta格式LLaMA模型目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出Hugging Face格式模型的目录")
    parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "13B", "70B"], help="模型大小")
    parser.add_argument("--chat_model", action="store_true", help="是否为chat模型")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载Meta模型
    meta_weights, params = load_meta_model(args.input_dir, args.model_size)
    
    # 创建配置
    config = get_model_config(params, args.model_size, args.chat_model)
    
    # 保存配置
    config.save_pretrained(args.output_dir)
    print(f"✓ 配置已保存到 {args.output_dir}/config.json")
    
    # 转换权重
    hf_state_dict = convert_weights(meta_weights, config)
    
    # 创建HF模型并加载权重
    model = LlamaForCausalLM(config)
    
    # 尝试加载状态字典
    try:
        missing_keys, unexpected_keys = model.load_state_dict(hf_state_dict, strict=False)
        if missing_keys:
            print(f"警告: 缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"警告: 意外的键: {unexpected_keys}")
    except Exception as e:
        print(f"错误: 无法加载状态字典: {e}")
        # 尝试手动设置权重
        for name, param in model.named_parameters():
            if name in hf_state_dict:
                param.data.copy_(hf_state_dict[name])
    
    # 保存模型
    print("保存转换后的模型...")
    model.save_pretrained(args.output_dir)
    print(f"✓ 模型已保存到 {args.output_dir}")
    
    # 准备分词器
    prepare_tokenizer(args.input_dir, args.output_dir, args.chat_model)
    
    # 验证
    print("\n验证转换结果...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        
        print("✓ 成功加载转换后的模型和分词器")
        print("转换完成!")
    except Exception as e:
        print(f"验证失败: {e}")
        print("转换可能不完整，请检查错误信息")

if __name__ == "__main__":
    main()
EOF

echo "✓ 转换脚本已创建"
echo ""

# 运行转换脚本
echo "步骤4: 运行转换脚本..."
python "$TEMP_DIR/convert.py" --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --model_size "$MODEL_SIZE" $([ "$CHAT_MODEL" = true ] && echo "--chat_model")

# 检查转换是否成功
if [ $? -ne 0 ]; then
    echo "错误: 转换失败"
    exit 1
fi

# 验证转换结果
echo ""
echo "步骤5: 验证转换结果..."

# 检查必要的文件
REQUIRED_FILES=("config.json" "pytorch_model.bin" "tokenizer.model" "tokenizer_config.json" "special_tokens_map.json")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$OUTPUT_DIR/$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "警告: 以下文件缺失:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo "转换可能不完整"
else
    echo "✓ 所有必要文件都已转换"
fi

# 打印概要
echo ""
echo "=================================================="
echo "  转换完成"
echo "=================================================="
echo ""
echo "Meta格式模型: $INPUT_DIR"
echo "Hugging Face模型: $OUTPUT_DIR"
echo ""
echo "现在您可以使用以下命令更新配置文件中的模型路径:"
echo "  ./fix_model_path.sh --model_path \"$OUTPUT_DIR\""
echo ""
echo "然后运行微调:"
echo "  bash run_finetune.sh --local_model"
echo ""

# 检查磁盘占用
INPUT_SIZE=$(du -sh "$INPUT_DIR" | cut -f1)
OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
echo "磁盘占用情况:"
echo "  - 原始Meta格式: $INPUT_SIZE"
echo "  - 转换后HF格式: $OUTPUT_SIZE"
echo "" 