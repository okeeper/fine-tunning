#!/bin/bash

echo "=================================================="
echo "  PyTorch 环境修复工具"
echo "  用于解决依赖冲突和 CUDA 兼容性问题"
echo "=================================================="
echo ""

# 设置日志颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 命令未找到，请先安装"
        return 1
    fi
    return 0
}

# 检查当前环境
check_environment() {
    log_info "正在检查当前环境..."
    
    # 检查 Python 版本
    if check_command python; then
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
        log_info "Python 版本: $PYTHON_VERSION"
    else
        log_error "未找到 Python，请先安装"
        exit 1
    fi
    
    # 检查 pip 版本
    if check_command pip; then
        PIP_VERSION=$(pip --version | awk '{print $2}')
        log_info "pip 版本: $PIP_VERSION"
    else
        log_error "未找到 pip，请先安装"
        exit 1
    fi
    
    # 检查 conda
    if command -v conda &> /dev/null; then
        CONDA_VERSION=$(conda --version | awk '{print $2}')
        log_info "Conda 版本: $CONDA_VERSION"
        CONDA_AVAILABLE=true
    else
        log_info "未找到 Conda，将使用 pip 进行包管理"
        CONDA_AVAILABLE=false
    fi
    
    # 检查当前环境
    if [ "$CONDA_AVAILABLE" = true ]; then
        CURRENT_ENV=$(conda info --envs | grep "*" | awk '{print $1}')
        log_info "当前 Conda 环境: $CURRENT_ENV"
    fi
    
    # 检查 PyTorch 版本
    if python -c "import torch" &> /dev/null; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        log_info "PyTorch 版本: $TORCH_VERSION"
        
        # 检查 CUDA 可用性
        CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
        if [ "$CUDA_AVAILABLE" = "True" ]; then
            CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
            log_info "CUDA 可用，版本: $CUDA_VERSION"
        else
            log_warning "CUDA 不可用"
            
            # 检查是否有 NVIDIA 驱动
            if command -v nvidia-smi &> /dev/null; then
                DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
                log_info "NVIDIA 驱动版本: $DRIVER_VERSION"
                
                # 计算最大兼容的 CUDA 版本
                DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
                if [ $DRIVER_MAJOR -ge 520 ]; then
                    MAX_CUDA="12.1"
                elif [ $DRIVER_MAJOR -ge 510 ]; then
                    MAX_CUDA="11.8"
                elif [ $DRIVER_MAJOR -ge 470 ]; then
                    MAX_CUDA="11.4"
                elif [ $DRIVER_MAJOR -ge 450 ]; then
                    MAX_CUDA="11.0"
                elif [ $DRIVER_MAJOR -ge 440 ]; then
                    MAX_CUDA="10.2"
                else
                    MAX_CUDA="10.0"
                fi
                log_info "驱动支持的最大 CUDA 版本: $MAX_CUDA"
            else
                log_warning "未找到 NVIDIA 驱动"
                MAX_CUDA="CPU"
            fi
        fi
    else
        log_warning "未安装 PyTorch"
        TORCH_VERSION="未安装"
        
        # 检查是否有 NVIDIA 驱动
        if command -v nvidia-smi &> /dev/null; then
            DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
            log_info "NVIDIA 驱动版本: $DRIVER_VERSION"
            
            # 计算最大兼容的 CUDA 版本
            DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
            if [ $DRIVER_MAJOR -ge 520 ]; then
                MAX_CUDA="12.1"
            elif [ $DRIVER_MAJOR -ge 510 ]; then
                MAX_CUDA="11.8"
            elif [ $DRIVER_MAJOR -ge 470 ]; then
                MAX_CUDA="11.4"
            elif [ $DRIVER_MAJOR -ge 450 ]; then
                MAX_CUDA="11.0"
            elif [ $DRIVER_MAJOR -ge 440 ]; then
                MAX_CUDA="10.2"
            else
                MAX_CUDA="10.0"
            fi
            log_info "驱动支持的最大 CUDA 版本: $MAX_CUDA"
        else
            log_warning "未找到 NVIDIA 驱动"
            MAX_CUDA="CPU"
        fi
    fi
    
    # 检查依赖包
    check_dependency() {
        if python -c "import $1" &> /dev/null; then
            VERSION=$(python -c "import $1; print($1.__version__)")
            log_info "$1 版本: $VERSION"
            return 0
        else
            log_warning "未安装 $1"
            return 1
        fi
    }
    
    check_dependency transformers
    check_dependency datasets
    check_dependency peft
    check_dependency accelerate
    check_dependency bitsandbytes
    check_dependency wandb
    
    echo ""
    log_info "环境检查完成"
    echo ""
}

# 显示修复选项
show_options() {
    echo "=================================================="
    echo "  可用修复选项"
    echo "=================================================="
    echo ""
    echo "1. 降级依赖包，使其与当前 PyTorch 版本兼容"
    echo "   - 适合不想更改 PyTorch 版本的情况"
    echo "   - 将安装与 PyTorch $TORCH_VERSION 兼容的依赖包版本"
    echo ""
    echo "2. 安装 CPU 版本的 PyTorch"
    echo "   - 适合没有 GPU 或驱动问题无法解决的情况"
    echo "   - 将安装最新的 CPU 版本 PyTorch 及兼容依赖"
    echo ""
    echo "3. 安装 CUDA 兼容版本的 PyTorch"
    echo "   - 根据您的驱动版本，自动选择兼容的 CUDA 版本"
    echo "   - 最大兼容 CUDA 版本: $MAX_CUDA"
    echo ""
    echo "4. 创建全新的 Conda 环境（推荐）"
    echo "   - 完全重建环境，确保所有依赖一致"
    echo "   - 将创建新环境并安装所有必要的包"
    echo ""
    echo "5. 退出"
    echo ""
    
    read -p "请选择修复选项 (1-5): " OPTION
    echo ""
}

# 降级依赖包
downgrade_dependencies() {
    log_info "正在降级依赖包..."
    
    # 获取 PyTorch 版本的主要部分（例如 1.12 而不是 1.12.1+cu113）
    TORCH_MAJOR=$(echo $TORCH_VERSION | cut -d. -f1)
    TORCH_MINOR=$(echo $TORCH_VERSION | cut -d. -f2)
    
    # 根据 PyTorch 版本选择兼容的依赖版本
    if [ $TORCH_MAJOR -eq 1 ] && [ $TORCH_MINOR -lt 13 ]; then
        # PyTorch < 1.13
        log_info "为 PyTorch $TORCH_MAJOR.$TORCH_MINOR 安装兼容的依赖包"
        pip uninstall -y accelerate
        pip install accelerate==0.20.3
        
        pip uninstall -y bitsandbytes bitsandbytes-cuda113
        pip install bitsandbytes-cuda113==0.27.2 || pip install bitsandbytes==0.27.2
        
        pip uninstall -y peft
        pip install peft==0.4.0
        
        pip install transformers==4.30.2
        
    elif [ $TORCH_MAJOR -eq 1 ] && [ $TORCH_MINOR -ge 13 ] && [ $TORCH_MINOR -lt 0 ]; then
        # PyTorch >= 1.13 && < 2.0
        log_info "为 PyTorch $TORCH_MAJOR.$TORCH_MINOR 安装兼容的依赖包"
        pip uninstall -y accelerate
        pip install accelerate==0.23.0
        
        pip uninstall -y bitsandbytes bitsandbytes-cuda113
        pip install bitsandbytes==0.39.1
        
        pip uninstall -y peft
        pip install peft==0.6.0
        
        pip install transformers==4.34.0
        
    else
        # PyTorch >= 2.0 或 CPU 版本
        log_info "为 PyTorch $TORCH_VERSION 安装最新的依赖包"
        pip install accelerate
        pip install bitsandbytes
        pip install peft
        pip install transformers
    fi
    
    pip install datasets
    pip install wandb
    
    log_success "依赖包降级完成"
}

# 安装 CPU 版本的 PyTorch
install_cpu_pytorch() {
    log_info "正在安装 CPU 版本的 PyTorch..."
    
    # 卸载现有的 PyTorch
    log_info "卸载现有的 PyTorch..."
    pip uninstall -y torch torchvision torchaudio
    pip uninstall -y bitsandbytes bitsandbytes-cuda* 
    
    # 安装 CPU 版本的 PyTorch
    log_info "安装 CPU 版本的 PyTorch 2.0.1..."
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
    
    # 安装兼容的依赖包
    log_info "安装兼容的依赖包..."
    pip install transformers==4.34.0
    pip install datasets
    pip install bitsandbytes-cpu
    pip install accelerate==0.23.0
    pip install peft==0.6.0
    pip install wandb
    
    log_success "CPU 版本的 PyTorch 安装完成"
}

# 安装 CUDA 兼容版本的 PyTorch
install_cuda_pytorch() {
    log_info "正在安装 CUDA 兼容版本的 PyTorch..."
    
    # 卸载现有的 PyTorch
    log_info "卸载现有的 PyTorch..."
    pip uninstall -y torch torchvision torchaudio
    pip uninstall -y bitsandbytes bitsandbytes-cuda*
    
    # 根据最大兼容的 CUDA 版本安装 PyTorch
    case $MAX_CUDA in
        "12.1")
            log_info "安装 PyTorch 2.1.0 (CUDA 12.1)..."
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
            ;;
        "11.8"|"11.7"|"11.6")
            log_info "安装 PyTorch 2.0.1 (CUDA 11.7)..."
            pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
            ;;
        "11.5"|"11.4"|"11.3")
            log_info "安装 PyTorch 1.13.1 (CUDA 11.6)..."
            pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu116
            ;;
        "11.2"|"11.1"|"11.0")
            log_info "安装 PyTorch 1.10.1 (CUDA 11.3)..."
            pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 --extra-index-url https://download.pytorch.org/whl/cu113
            ;;
        "10.2")
            log_info "安装 PyTorch 1.9.1 (CUDA 10.2)..."
            pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 --extra-index-url https://download.pytorch.org/whl/cu102
            ;;
        *)
            log_warning "无法确定兼容的 CUDA 版本，回退到 CPU 版本..."
            install_cpu_pytorch
            return
            ;;
    esac
    
    # 安装兼容的依赖包
    log_info "安装兼容的依赖包..."
    if [[ $MAX_CUDA == "12.1" || $MAX_CUDA == "11.8" || $MAX_CUDA == "11.7" || $MAX_CUDA == "11.6" ]]; then
        # 较新版本的 PyTorch
        pip install transformers
        pip install datasets
        pip install bitsandbytes
        pip install accelerate
        pip install peft
    else
        # 较旧版本的 PyTorch
        pip install transformers==4.30.2
        pip install datasets
        
        if [[ $MAX_CUDA == "11.5" || $MAX_CUDA == "11.4" || $MAX_CUDA == "11.3" ]]; then
            pip install bitsandbytes-cuda113==0.27.2 || pip install bitsandbytes==0.27.2
        else
            pip install bitsandbytes==0.27.2
        fi
        
        pip install accelerate==0.20.3
        pip install peft==0.4.0
    fi
    
    pip install wandb
    
    log_success "CUDA 兼容版本的 PyTorch 安装完成"
}

# 创建全新的 Conda 环境
create_conda_env() {
    if [ "$CONDA_AVAILABLE" != true ]; then
        log_error "未找到 Conda，无法创建新环境"
        return 1
    fi
    
    # 获取新环境名称
    read -p "请输入新环境名称 [llama_ft_new]: " ENV_NAME
    ENV_NAME=${ENV_NAME:-llama_ft_new}
    
    log_info "正在创建新的 Conda 环境: $ENV_NAME..."
    
    # 创建新环境
    conda create -n $ENV_NAME python=3.10 -y
    
    # 激活新环境
    log_info "激活环境 $ENV_NAME..."
    
    # 这里不能直接使用 conda activate，因为脚本是在独立的 shell 中运行的
    # 我们需要生成激活命令，让用户手动执行
    echo ""
    log_warning "需要手动激活新环境。请复制并执行以下命令:"
    echo ""
    echo "conda activate $ENV_NAME"
    echo ""
    
    # 生成安装命令
    echo "然后运行以下命令安装依赖:"
    echo ""
    
    if [ "$MAX_CUDA" = "CPU" ]; then
        # CPU 版本
        echo "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu"
        echo "pip install transformers==4.34.0 datasets bitsandbytes-cpu accelerate==0.23.0 peft==0.6.0 wandb"
    else
        # 根据 CUDA 版本选择 PyTorch
        case $MAX_CUDA in
            "12.1")
                echo "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121"
                echo "pip install transformers datasets bitsandbytes accelerate peft wandb"
                ;;
            "11.8"|"11.7"|"11.6")
                echo "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117"
                echo "pip install transformers datasets bitsandbytes accelerate peft wandb"
                ;;
            "11.5"|"11.4"|"11.3")
                echo "pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu116"
                echo "pip install transformers==4.30.2 datasets bitsandbytes-cuda113==0.27.2 accelerate==0.20.3 peft==0.4.0 wandb"
                ;;
            "11.2"|"11.1"|"11.0")
                echo "pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 --extra-index-url https://download.pytorch.org/whl/cu113"
                echo "pip install transformers==4.30.2 datasets bitsandbytes==0.27.2 accelerate==0.20.3 peft==0.4.0 wandb"
                ;;
            "10.2")
                echo "pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 --extra-index-url https://download.pytorch.org/whl/cu102"
                echo "pip install transformers==4.30.2 datasets bitsandbytes==0.27.2 accelerate==0.20.3 peft==0.4.0 wandb"
                ;;
            *)
                echo "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu"
                echo "pip install transformers==4.34.0 datasets bitsandbytes-cpu accelerate==0.23.0 peft==0.6.0 wandb"
                ;;
        esac
    fi
    
    echo ""
    log_success "Conda 环境创建完成。请按上述步骤操作。"
}

# 主函数
main() {
    # 检查当前环境
    check_environment
    
    # 显示修复选项
    show_options
    
    # 执行选择的修复选项
    case $OPTION in
        1)
            downgrade_dependencies
            ;;
        2)
            install_cpu_pytorch
            ;;
        3)
            install_cuda_pytorch
            ;;
        4)
            create_conda_env
            ;;
        5)
            log_info "退出"
            exit 0
            ;;
        *)
            log_error "无效的选项: $OPTION"
            exit 1
            ;;
    esac
    
    # 验证修复结果
    echo ""
    log_info "验证环境..."
    python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available(), ', CUDA 版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
    pip list | grep -E "torch|accelerate|bitsandbytes|peft|transformers"
    
    echo ""
    log_success "修复完成。请尝试重新运行您的脚本。"
    echo ""
    echo "推荐的运行命令:"
    echo "  bash run_finetune.sh --use_cpu --max_samples 1000  # CPU 模式"
    if [ "$CUDA_AVAILABLE" = "True" ] || [ "$MAX_CUDA" != "CPU" ]; then
        echo "  bash run_finetune.sh  # GPU 模式"
    fi
    echo ""
}

# 执行主函数
main