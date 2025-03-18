#!/bin/bash

echo "=================================================="
echo "  PyTorch 与 CUDA 兼容性修复"
echo "=================================================="
echo ""

# 检查当前环境
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python 版本: $PYTHON_VERSION"

# 检查CUDA版本
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "检测到CUDA版本: $CUDA_VERSION"
else
    echo "未检测到CUDA，将使用CPU版本"
    CUDA_VERSION="none"
fi

# 检查是否在conda环境中
if [ -z "$CONDA_PREFIX" ]; then
    echo "警告: 未检测到激活的conda环境"
    echo "建议在conda环境中运行此脚本"
    echo ""
    read -p "是否继续? (y/n): " continue_without_conda
    if [[ ! "$continue_without_conda" =~ ^[Yy]$ ]]; then
        echo "操作取消。请创建并激活conda环境后再试"
        exit 1
    fi
fi

echo "选择安装方法:"
echo "1) 使用conda安装 (推荐)"
echo "2) 使用pip安装"
echo "3) 安装CPU版本 (无需CUDA)"
echo "4) 创建新的conda环境"
read -p "请选择 [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "步骤1: 使用conda安装PyTorch..."
        
        if [[ "$CUDA_VERSION" == "10.2" ]]; then
            echo "安装与CUDA 10.2兼容的PyTorch 1.10.1..."
            conda install -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 cudatoolkit=10.2 -c pytorch
        elif [[ "$CUDA_VERSION" == "11.3" || "$CUDA_VERSION" == "11.4" || "$CUDA_VERSION" == "11.5" || "$CUDA_VERSION" == "11.6" ]]; then
            echo "安装与CUDA 11.3/11.6兼容的PyTorch 1.12.1..."
            conda install -y pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch
        elif [[ "$CUDA_VERSION" == "11.7" || "$CUDA_VERSION" == "11.8" ]]; then
            echo "安装与CUDA 11.7/11.8兼容的PyTorch 1.13.1..."
            conda install -y pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
        elif [[ "$CUDA_VERSION" == "12."* ]]; then
            echo "安装与CUDA 12.x兼容的PyTorch 2.0.0+..."
            conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
        else
            echo "未能识别CUDA版本或没有CUDA，将安装CPU版本的PyTorch 1.10.1..."
            conda install -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 cpuonly -c pytorch
        fi
        ;;
    2)
        echo ""
        echo "步骤1: 使用pip安装PyTorch..."
        
        # 卸载现有PyTorch
        pip uninstall -y torch torchvision torchaudio
        
        if [[ "$CUDA_VERSION" == "10.2" ]]; then
            echo "安装与CUDA 10.2兼容的PyTorch 1.10.1..."
            pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
        elif [[ "$CUDA_VERSION" == "11.3" || "$CUDA_VERSION" == "11.4" || "$CUDA_VERSION" == "11.5" || "$CUDA_VERSION" == "11.6" ]]; then
            echo "安装与CUDA 11.3/11.6兼容的PyTorch 1.12.1..."
            pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
        elif [[ "$CUDA_VERSION" == "11.7" || "$CUDA_VERSION" == "11.8" ]]; then
            echo "安装与CUDA 11.7/11.8兼容的PyTorch 1.13.1..."
            pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
        elif [[ "$CUDA_VERSION" == "12."* ]]; then
            echo "安装与CUDA 12.x兼容的PyTorch 2.0.0+..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        else
            echo "未能识别CUDA版本或没有CUDA，将安装CPU版本的PyTorch 1.10.1..."
            pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 --index-url https://download.pytorch.org/whl/cpu
        fi
        ;;
    3)
        echo ""
        echo "步骤1: 安装CPU版本的PyTorch..."
        
        # 卸载现有PyTorch
        pip uninstall -y torch torchvision torchaudio
        
        # 安装CPU版本
        echo "安装CPU版本的PyTorch 1.10.1..."
        pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 --index-url https://download.pytorch.org/whl/cpu
        ;;
    4)
        echo ""
        echo "步骤1: 创建新的conda环境..."
        read -p "输入新环境名称 [默认: llama_cuda_env]: " env_name
        env_name=${env_name:-llama_cuda_env}
        
        if [[ "$CUDA_VERSION" == "10.2" ]]; then
            echo "创建与CUDA 10.2兼容的新环境 $env_name..."
            conda create -n $env_name python=3.8 -y
            conda install -n $env_name -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 cudatoolkit=10.2 -c pytorch
        elif [[ "$CUDA_VERSION" == "11.3" || "$CUDA_VERSION" == "11.4" || "$CUDA_VERSION" == "11.5" || "$CUDA_VERSION" == "11.6" ]]; then
            echo "创建与CUDA 11.3/11.6兼容的新环境 $env_name..."
            conda create -n $env_name python=3.8 -y
            conda install -n $env_name -y pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch
        elif [[ "$CUDA_VERSION" == "11.7" || "$CUDA_VERSION" == "11.8" ]]; then
            echo "创建与CUDA 11.7/11.8兼容的新环境 $env_name..."
            conda create -n $env_name python=3.9 -y
            conda install -n $env_name -y pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
        elif [[ "$CUDA_VERSION" == "12."* ]]; then
            echo "创建与CUDA 12.x兼容的新环境 $env_name..."
            conda create -n $env_name python=3.10 -y
            conda install -n $env_name -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
        else
            echo "未能识别CUDA版本或没有CUDA，将创建CPU版本的环境 $env_name..."
            conda create -n $env_name python=3.8 -y
            conda install -n $env_name -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 cpuonly -c pytorch
        fi
        
        # 安装其他依赖
        echo ""
        echo "安装其他依赖..."
        conda run -n $env_name pip install transformers accelerate peft trl
        conda run -n $env_name pip install datasets wandb
        
        echo ""
        echo "新环境 $env_name 已创建并安装了兼容的PyTorch。"
        echo "请使用以下命令切换到新环境:"
        echo "  conda activate $env_name"
        exit 0
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

# 安装其他依赖
echo ""
echo "步骤2: 安装其他依赖..."
pip install -U transformers accelerate peft trl datasets wandb

# 如果是CUDA 10.2，安装兼容的bitsandbytes版本
if [[ "$CUDA_VERSION" == "10.2" ]]; then
    echo ""
    echo "步骤3: 安装与CUDA 10.2兼容的bitsandbytes版本..."
    pip uninstall -y bitsandbytes
    pip install bitsandbytes==0.38.1
fi

# 验证安装
echo ""
echo "步骤4: 验证PyTorch安装..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
if torch.cuda.is_available():
    print(f'GPU型号: {torch.cuda.get_device_name(0)}')
    print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"

# 更新配置文件
echo ""
echo "步骤5: 更新配置文件以提高兼容性..."
CONFIG_FILE="config/finetune_config.json"

if [ -f "$CONFIG_FILE" ]; then
    # 备份配置文件
    cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
    echo "已备份配置文件到 ${CONFIG_FILE}.backup"
    
    # 更新配置
    python -c "
import json
import os

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = json.load(f)
    
    # 根据环境调整训练参数
    if 'training_args' in config:
        # 对于较旧的CUDA版本或CPU模式，禁用fp16和bf16
        if '$CUDA_VERSION' == '10.2' or '$CUDA_VERSION' == 'none':
            config['training_args']['fp16'] = False
            config['training_args']['bf16'] = False
            # 调小批次大小
            config['training_args']['per_device_train_batch_size'] = 1
            config['training_args']['gradient_accumulation_steps'] = 16
            # 降低学习率
            config['training_args']['learning_rate'] = 1e-5
    
    # 保存更新后的配置
    with open('$CONFIG_FILE', 'w') as f:
        json.dump(config, f, indent=4)
    print('成功更新配置文件以提高兼容性')
except Exception as e:
    print(f'更新配置文件时出错: {e}')
"
else
    echo "警告: 找不到配置文件 $CONFIG_FILE"
fi

echo ""
echo "=================================================="
echo "  PyTorch 安装和配置完成"
echo "=================================================="
echo ""
echo "推荐运行命令:"
if [[ "$CUDA_VERSION" == "10.2" ]]; then
    echo "bash run_finetune.sh --local_model --max_samples 1000"
    echo ""
    echo "如果仍有问题，可以尝试CPU模式:"
    echo "bash run_finetune.sh --use_cpu --local_model --max_samples 1000"
elif [[ "$CUDA_VERSION" == "none" ]]; then
    echo "bash run_finetune.sh --use_cpu --local_model --max_samples 1000"
else
    echo "bash run_finetune.sh --local_model"
fi
echo ""