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
import traceback
import argparse
import gc
from tqdm import tqdm

def print_memory_usage():
    """打印系统内存使用情况"""
    try:
        import psutil
        print("系统内存情况:")
        print(f"总内存: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        print(f"可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        print(f"使用内存: {psutil.virtual_memory().used / (1024**3):.2f} GB")
        print(f"内存使用率: {psutil.virtual_memory().percent}%")
    except ImportError:
        print("未安装psutil，无法显示内存使用情况")

def convert_meta_to_hf(input_dir, output_dir, vocab_size=32000, model_size="7B", chat_model=True):
    """将Meta格式的LLaMA模型转换为Hugging Face格式"""
    try:
        print(f"正在将Meta格式模型从 {input_dir} 转换到 {output_dir}...")
        print(f"词表大小: {vocab_size}, 模型大小: {model_size}, 聊天模型: {chat_model}")
        
        # 打印内存使用情况
        print_memory_usage()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型参数
        params_path = os.path.join(input_dir, "params.json")
        if not os.path.exists(params_path):
            print(f"错误: 找不到参数文件 {params_path}")
            sys.exit(1)
        
        print("读取模型参数文件...")
        with open(params_path, "r") as f:
            params = json.load(f)
        
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
            print(f"使用params.json中的参数...")
            dim = params.get("dim", 4096)
            n_layers = params.get("n_layers", 32)
            n_heads = params.get("n_heads", 32)
            intermediate_size = int(dim * 2.7)  # 近似值
        
        print(f"模型参数: dim={dim}, layers={n_layers}, heads={n_heads}, intermediate_size={intermediate_size}")
        print(f"使用词表大小: {vocab_size}")
        
        # 创建配置文件
        print("创建模型配置...")
        config = {
            "architectures": [
                "LlamaForCausalLM"
            ],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": dim,
            "initializer_range": 0.02,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": params.get("max_seq_len", 4096),
            "model_type": "llama",
            "num_attention_heads": n_heads,
            "num_hidden_layers": n_layers,
            "pad_token_id": 0,
            "rms_norm_eps": params.get("norm_eps", 1e-5),
            "tie_word_embeddings": False,
            "transformers_version": "4.30.2",
            "use_cache": True,
            "vocab_size": vocab_size
        }
        
        # 如果是聊天模型，添加聊天模板
        if chat_model:
            config["chat_template"] = "{% for message in messages %}\n{% if message['role'] == 'system' %}\n<s>[INST] <<SYS>>\n{{ message['content'] }}\n<</SYS>>\n\n{% elif message['role'] == 'user' %}\n{% if loop.index > 1 %}</s>{% endif %}\n<s>[INST] {{ message['content'] }} [/INST]\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] }}\n{% endif %}\n{% endfor %}"
        
        # 保存配置文件
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # 构建权重映射字典
        print("创建权重映射...")
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
        
        # 检查是否有多个分片文件
        consolidated_files = [f for f in os.listdir(input_dir) if f.startswith("consolidated.") and f.endswith(".pth")]
        consolidated_files.sort()
        
        # 创建索引文件
        index_data = {"metadata": {"total_size": 0}, "weight_map": {}}
        
        # 定义分片大小 (每个分片最大约2GB)
        shard_size = 2 * 1024 * 1024 * 1024
        
        # 处理所有分片
        current_shard_size = 0
        current_shard_tensors = {}
        shard_index = 0
        
        # 循环处理每个Meta模型分片
        for meta_shard_file in consolidated_files:
            print(f"处理Meta模型分片: {meta_shard_file}")
            
            meta_shard_path = os.path.join(input_dir, meta_shard_file)
            
            # 加载Meta模型分片
            print(f"加载权重文件: {meta_shard_path}")
            try:
                meta_weights = torch.load(meta_shard_path, map_location="cpu")
            except Exception as e:
                print(f"错误: 无法加载模型权重: {e}")
                sys.exit(1)
            
            # 处理这个分片中的每个权重
            print(f"处理 {len(meta_weights)} 个权重参数...")
            
            # 进度条
            for meta_name, meta_tensor in tqdm(meta_weights.items(), desc="转换权重"):
                if meta_name in weight_map:
                    hf_name = weight_map[meta_name]
                    
                    # 特殊处理嵌入层
                    if meta_name == "tok_embeddings.weight":
                        meta_vocab_size = meta_tensor.shape[0]
                        
                        if meta_vocab_size != vocab_size:
                            if meta_vocab_size < vocab_size:
                                print(f"将embed_tokens.weight从 {meta_vocab_size} 填充到 {vocab_size} 词")
                                padding = torch.zeros(
                                    (vocab_size - meta_vocab_size, meta_tensor.shape[1]),
                                    dtype=meta_tensor.dtype
                                )
                                meta_tensor = torch.cat([meta_tensor, padding], dim=0)
                            else:
                                print(f"将embed_tokens.weight从 {meta_vocab_size} 截断到 {vocab_size} 词")
                                meta_tensor = meta_tensor[:vocab_size, :]
                    
                    # 计算张量大小
                    tensor_size = meta_tensor.numel() * meta_tensor.element_size()
                    
                    # 检查是否需要创建新的分片
                    if current_shard_size + tensor_size > shard_size and current_shard_tensors:
                        # 保存当前分片
                        shard_filename = f"pytorch_model-{shard_index:05d}-of-00010.bin"
                        shard_path = os.path.join(output_dir, shard_filename)
                        print(f"保存分片 {shard_path} (大小: {current_shard_size / 1024**2:.2f} MB)")
                        torch.save(current_shard_tensors, shard_path)
                        
                        # 更新索引
                        for tensor_name in current_shard_tensors.keys():
                            index_data["weight_map"][tensor_name] = shard_filename
                        
                        # 更新总大小
                        index_data["metadata"]["total_size"] += current_shard_size
                        
                        # 重置当前分片
                        current_shard_tensors = {}
                        current_shard_size = 0
                        shard_index += 1
                    
                    # 添加到当前分片
                    current_shard_tensors[hf_name] = meta_tensor
                    current_shard_size += tensor_size
                    
                # 清理内存
                if len(current_shard_tensors) > 10:  # 每处理10个张量就清理一次内存
                    gc.collect()
            
            # 清理分片内存
            del meta_weights
            gc.collect()
        
        # 保存最后一个分片
        if current_shard_tensors:
            shard_filename = f"pytorch_model-{shard_index:05d}-of-00010.bin"
            shard_path = os.path.join(output_dir, shard_filename)
            print(f"保存最后一个分片 {shard_path} (大小: {current_shard_size / 1024**2:.2f} MB)")
            torch.save(current_shard_tensors, shard_path)
            
            # 更新索引
            for tensor_name in current_shard_tensors.keys():
                index_data["weight_map"][tensor_name] = shard_filename
            
            # 更新总大小
            index_data["metadata"]["total_size"] += current_shard_size
        
        # 保存索引文件
        index_data["metadata"]["total_shards"] = shard_index + 1
        
        with open(os.path.join(output_dir, "pytorch_model.bin.index.json"), "w") as f:
            json.dump(index_data, f, indent=2)
        
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
        
        # 创建generation_config.json
        generation_config = {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "transformers_version": "4.30.2"
        }
        
        with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
            json.dump(generation_config, f, indent=2)
        
        print("转换完成！")
        print(f"Hugging Face格式的模型已保存到: {output_dir}")
        print("可以使用以下方式加载模型:")
        print(f"from transformers import AutoTokenizer, AutoModelForCausalLM")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
        print(f"model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
        
        return True
    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)

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

# 安装必要的Python库
log_info "安装必要的Python库..."
pip install -q psutil tqdm

# 运行Python脚本
if PYTHONPATH="$PYTHONPATH:$(dirname "$INPUT_DIR")" python -u "$CONVERT_SCRIPT" \
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