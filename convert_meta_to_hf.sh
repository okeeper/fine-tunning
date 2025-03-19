#!/bin/bash

# Meta格式LLaMA模型转换为Hugging Face格式的脚本
# 优化版：支持自动检测词表大小、参数验证和详细日志

set -e  # 遇到错误立即退出

# 文字颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 重置颜色

# 显示帮助信息
function show_help {
    echo -e "${BLUE}Meta格式LLaMA模型转换为Hugging Face格式${NC}"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --input_dir DIR       Meta格式模型的输入目录 (必需)"
    echo "  --output_dir DIR      Hugging Face格式模型的输出目录 (必需)"
    echo "  --vocab_size SIZE     词表大小 (默认: 自动检测)"
    echo "  --chat_model          是否为聊天模型 (默认: 是)"
    echo "  --model_size SIZE     模型大小 [7B, 13B, 70B] (默认: 7B)"
    echo "  --help                显示此帮助信息"
    echo ""
    echo "例子:"
    echo "  $0 --input_dir /opt/llama/Llama-2-7b-chat --output_dir /opt/llama/Llama-2-7b-chat-hf"
    echo "  $0 --input_dir /opt/llama/Llama-2-7b-chat --output_dir /opt/llama/Llama-2-7b-chat-hf --vocab_size 32000"
    echo ""
}

# 显示错误并退出
function error_exit {
    echo -e "${RED}错误: $1${NC}" >&2
    exit 1
}

# 显示警告但继续执行
function warning {
    echo -e "${YELLOW}警告: $1${NC}" >&2
}

# 检查路径存在
function check_path_exists {
    if [ ! -e "$1" ]; then
        error_exit "路径不存在: $1"
    fi
}

# 检查文件存在
function check_file_exists {
    if [ ! -f "$1" ]; then
        error_exit "文件不存在: $1"
    fi
}

# 检查目录存在
function check_dir_exists {
    if [ ! -d "$1" ]; then
        error_exit "目录不存在: $1"
    fi
}

# 从params.json检测词表大小
function detect_vocab_size_from_params {
    local params_file="$1"
    check_file_exists "$params_file"
    
    if command -v jq &> /dev/null; then
        # 使用jq解析JSON
        vocab_size=$(jq -r '.vocab_size // .n_vocab // .dim_vocab // empty' "$params_file")
        if [ -z "$vocab_size" ] || [ "$vocab_size" = "null" ] || [ "$vocab_size" -le 0 ]; then
            return 1
        fi
        echo $vocab_size
        return 0
    else
        # 如果没有jq，使用grep和sed
        vocab_size=$(grep -o '"vocab_size":[0-9]*\|"n_vocab":[0-9]*\|"dim_vocab":[0-9]*' "$params_file" | head -1 | sed -E 's/.*:([0-9]+).*/\1/')
        if [ -z "$vocab_size" ] || [ "$vocab_size" -le 0 ]; then
            return 1
        fi
        echo $vocab_size
        return 0
    fi
}

# 从tokenizer.model检测词表大小
function detect_vocab_size_from_tokenizer {
    local tokenizer_file="$1"
    check_file_exists "$tokenizer_file"
    
    # 创建临时Python脚本来检测词表大小
    local temp_script=$(mktemp)
    cat > "$temp_script" << 'EOF'
import sys
try:
    import sentencepiece as spm
    model = spm.SentencePieceProcessor()
    model.Load(sys.argv[1])
    print(model.GetPieceSize())
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    
    # 运行Python脚本
    if ! python "$temp_script" "$tokenizer_file" 2>/dev/null; then
        rm -f "$temp_script"
        return 1
    fi
    
    rm -f "$temp_script"
    return 0
}

# 输出信息
function log_info {
    echo -e "${GREEN}[INFO] $1${NC}"
}

# 解析命令行参数
INPUT_DIR=""
OUTPUT_DIR=""
VOCAB_SIZE=""
CHAT_MODEL=1
MODEL_SIZE="7B"

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
        --vocab_size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        --chat_model)
            CHAT_MODEL=1
            shift
            ;;
        --base_model)
            CHAT_MODEL=0
            shift
            ;;
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            error_exit "未知选项: $1"
            ;;
    esac
done

# 检查必需参数
if [ -z "$INPUT_DIR" ]; then
    error_exit "缺少必需参数: --input_dir"
fi

if [ -z "$OUTPUT_DIR" ]; then
    error_exit "缺少必需参数: --output_dir"
fi

# 检查输入目录
check_dir_exists "$INPUT_DIR"

# 检查必需文件
MODEL_FILE="$INPUT_DIR/consolidated.00.pth"
PARAMS_FILE="$INPUT_DIR/params.json"
TOKENIZER_FILE="$INPUT_DIR/tokenizer.model"

check_file_exists "$MODEL_FILE"
check_file_exists "$PARAMS_FILE"
check_file_exists "$TOKENIZER_FILE"

# 如果没有指定词表大小，尝试自动检测
if [ -z "$VOCAB_SIZE" ]; then
    log_info "尝试从模型文件自动检测词表大小..."
    
    # 首先尝试从params.json检测
    DETECTED_VOCAB_SIZE=$(detect_vocab_size_from_params "$PARAMS_FILE" || echo "")
    
    if [ -n "$DETECTED_VOCAB_SIZE" ] && [ "$DETECTED_VOCAB_SIZE" -gt 0 ]; then
        log_info "从params.json检测到词表大小: $DETECTED_VOCAB_SIZE"
        VOCAB_SIZE=$DETECTED_VOCAB_SIZE
    else
        # 尝试从tokenizer.model检测
        DETECTED_VOCAB_SIZE=$(detect_vocab_size_from_tokenizer "$TOKENIZER_FILE" || echo "")
        
        if [ -n "$DETECTED_VOCAB_SIZE" ] && [ "$DETECTED_VOCAB_SIZE" -gt 0 ]; then
            log_info "从tokenizer.model检测到词表大小: $DETECTED_VOCAB_SIZE"
            VOCAB_SIZE=$DETECTED_VOCAB_SIZE
        else
            # 如果都检测不到，使用默认值
            warning "无法自动检测词表大小，使用默认值 32000"
            VOCAB_SIZE=32000
        fi
    fi
fi

# 确保VOCAB_SIZE是正整数
if ! [[ "$VOCAB_SIZE" =~ ^[0-9]+$ ]] || [ "$VOCAB_SIZE" -le 0 ]; then
    error_exit "词表大小必须为正整数，检测到的值无效: $VOCAB_SIZE"
fi

# 创建转换脚本
CONVERT_SCRIPT=$(mktemp)
cat > "$CONVERT_SCRIPT" << 'EOF'
import os
import sys
import json
import torch
import argparse
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

def convert_meta_to_hf(input_dir, output_dir, vocab_size=32000, model_size="7B", chat_model=True):
    """将Meta格式的LLaMA模型转换为Hugging Face格式"""
    print(f"正在将Meta格式模型从 {input_dir} 转换到 {output_dir}...")
    print(f"词表大小: {vocab_size}, 模型大小: {model_size}, 聊天模型: {chat_model}")
    
    # 加载Meta格式的模型权重
    consolidated_path = os.path.join(input_dir, "consolidated.00.pth")
    if not os.path.exists(consolidated_path):
        print(f"错误: 找不到模型权重文件 {consolidated_path}")
        sys.exit(1)
    
    # 加载模型参数
    params_path = os.path.join(input_dir, "params.json")
    if not os.path.exists(params_path):
        print(f"错误: 找不到参数文件 {params_path}")
        sys.exit(1)
    
    with open(params_path, "r") as f:
        params = json.load(f)
    
    # 加载权重
    print("加载模型权重...")
    try:
        meta_weights = torch.load(consolidated_path, map_location="cpu")
    except Exception as e:
        print(f"错误: 加载模型权重失败: {e}")
        sys.exit(1)
    
    # 创建Hugging Face配置
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
        n_heads = 80
        intermediate_size = 28672
    else:
        # 从params获取
        dim = params.get("dim", 4096)
        n_layers = params.get("n_layers", 32)
        n_heads = params.get("n_heads", 32)
        intermediate_size = int(dim * 2.7)  # 近似值
    
    # 使用指定的词表大小
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=dim,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=params.get("max_seq_len", 4096),
        rms_norm_eps=params.get("norm_eps", 1e-5),
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    
    # 如果是聊天模型，添加聊天模板
    if chat_model:
        config.chat_template = "{% for message in messages %}\n{% if message['role'] == 'system' %}\n<s>[INST] <<SYS>>\n{{ message['content'] }}\n<</SYS>>\n\n{% elif message['role'] == 'user' %}\n{% if loop.index > 1 %}</s>{% endif %}\n<s>[INST] {{ message['content'] }} [/INST]\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] }}\n{% endif %}\n{% endfor %}"
    
    # 创建模型
    print("创建模型结构...")
    model = LlamaForCausalLM(config)
    
    # 构建权重映射字典
    print("转换权重格式...")
    weight_map = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    
    for i in range(n_layers):
        # 注意力层权重
        weight_map.update({
            f"layers.{i}.attention.wq.weight": f"model.layers.{i}.self_attn.q_proj.weight",
            f"layers.{i}.attention.wk.weight": f"model.layers.{i}.self_attn.k_proj.weight",
            f"layers.{i}.attention.wv.weight": f"model.layers.{i}.self_attn.v_proj.weight",
            f"layers.{i}.attention.wo.weight": f"model.layers.{i}.self_attn.o_proj.weight",
        })
        
        # FFN权重
        weight_map.update({
            f"layers.{i}.feed_forward.w1.weight": f"model.layers.{i}.mlp.gate_proj.weight",
            f"layers.{i}.feed_forward.w2.weight": f"model.layers.{i}.mlp.down_proj.weight",
            f"layers.{i}.feed_forward.w3.weight": f"model.layers.{i}.mlp.up_proj.weight",
        })
        
        # 层归一化权重
        weight_map.update({
            f"layers.{i}.attention_norm.weight": f"model.layers.{i}.input_layernorm.weight",
            f"layers.{i}.ffn_norm.weight": f"model.layers.{i}.post_attention_layernorm.weight",
        })
    
    # 转换权重
    new_state_dict = {}
    missing_keys = []
    shape_mismatch_keys = []
    
    # 检查嵌入层大小是否匹配
    embed_key = "tok_embeddings.weight"
    if embed_key in meta_weights:
        meta_embed = meta_weights[embed_key]
        meta_vocab_size = meta_embed.shape[0]
        
        if meta_vocab_size != vocab_size:
            print(f"警告: 词表大小不匹配 - Meta模型: {meta_vocab_size}, 配置: {vocab_size}")
            print(f"将调整embed_tokens.weight的大小以匹配配置中的词表大小")
    
    for meta_name, meta_tensor in meta_weights.items():
        if meta_name in weight_map:
            hf_name = weight_map[meta_name]
            
            # 特殊处理嵌入层，处理词表大小不匹配的情况
            if meta_name == "tok_embeddings.weight":
                meta_vocab_size = meta_tensor.shape[0]
                
                if meta_vocab_size != vocab_size:
                    if meta_vocab_size < vocab_size:
                        # 如果Meta模型词表更小，则填充
                        print(f"将embed_tokens.weight从 {meta_vocab_size} 填充到 {vocab_size} 词")
                        padding = torch.zeros(
                            (vocab_size - meta_vocab_size, meta_tensor.shape[1]),
                            dtype=meta_tensor.dtype
                        )
                        meta_tensor = torch.cat([meta_tensor, padding], dim=0)
                    else:
                        # 如果Meta模型词表更大，则截断
                        print(f"将embed_tokens.weight从 {meta_vocab_size} 截断到 {vocab_size} 词")
                        meta_tensor = meta_tensor[:vocab_size, :]
            
            new_state_dict[hf_name] = meta_tensor
        else:
            missing_keys.append(meta_name)
    
    # 报告转换统计
    print(f"转换了 {len(new_state_dict)} 个权重参数")
    if missing_keys:
        print(f"跳过了 {len(missing_keys)} 个未映射的参数")
        print(f"前5个未映射参数示例: {missing_keys[:5]}")
    
    # 加载转换后的权重
    print("加载转换后的权重到模型...")
    model.load_state_dict(new_state_dict, strict=False)
    
    # 保存模型
    print(f"保存模型到 {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    
    # 准备分词器
    print("准备分词器...")
    tokenizer_path = os.path.join(input_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        print(f"错误: 找不到分词器文件 {tokenizer_path}")
        sys.exit(1)
    
    # 复制分词器文件
    os.system(f"cp {tokenizer_path} {output_dir}/")
    
    # 创建tokenizer_config.json
    tokenizer_config = {
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
    
    # 如果是聊天模型，添加聊天模板
    if chat_model:
        tokenizer_config["chat_template"] = "{% for message in messages %}\n{% if message['role'] == 'system' %}\n<s>[INST] <<SYS>>\n{{ message['content'] }}\n<</SYS>>\n\n{% elif message['role'] == 'user' %}\n{% if loop.index > 1 %}</s>{% endif %}\n<s>[INST] {{ message['content'] }} [/INST]\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] }}\n{% endif %}\n{% endfor %}"
    
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # 创建special_tokens_map.json
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>"
    }
    
    with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens, f, indent=2)
    
    print("转换完成！")
    print(f"Hugging Face格式的模型已保存到: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Meta LLaMA format to Hugging Face format")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with Meta format model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for HF format model")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--model_size", type=str, default="7B", help="Model size (7B, 13B, 70B)")
    parser.add_argument("--chat_model", action="store_true", help="Whether the model is a chat model")
    
    args = parser.parse_args()
    convert_meta_to_hf(args.input_dir, args.output_dir, args.vocab_size, args.model_size, args.chat_model)
EOF

# 执行转换
log_info "开始转换模型..."
log_info "输入目录: $INPUT_DIR"
log_info "输出目录: $OUTPUT_DIR"
log_info "词表大小: $VOCAB_SIZE"
log_info "模型大小: $MODEL_SIZE"
log_info "是否为聊天模型: $([ $CHAT_MODEL -eq 1 ] && echo '是' || echo '否')"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行Python脚本
if python "$CONVERT_SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --vocab_size "$VOCAB_SIZE" \
    --model_size "$MODEL_SIZE" \
    $([ $CHAT_MODEL -eq 1 ] && echo "--chat_model"); then
    
    log_info "转换成功！"
    log_info "Hugging Face格式的模型已保存到: $OUTPUT_DIR"
    
    # 添加配置文件更新建议
    log_info "建议更新配置文件中的模型路径:"
    log_info "sed -i 's|\"model_name_or_path\": \"$INPUT_DIR\"|\"model_name_or_path\": \"$OUTPUT_DIR\"|g' config/finetune_config.json"
    
    # 添加运行指令建议
    log_info "您可以使用以下命令运行微调:"
    log_info "bash run_finetune.sh --local_model"
    
    # 清理临时文件
    rm -f "$CONVERT_SCRIPT"
    exit 0
else
    error_exit "转换失败！请检查日志获取详细信息。"
fi 