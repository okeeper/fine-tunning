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

# 解析命令行参数
OPTION=""
SHOW_HELP=false
AUTO_FIX=false

# 处理命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        --auto)
            AUTO_FIX=true
            shift
            ;;
        --downgrade|--option=1|--option1|-1)
            OPTION=1
            shift
            ;;
        --cpu|--option=2|--option2|-2)
            OPTION=2
            shift
            ;;
        --cuda|--option=3|--option3|-3)
            OPTION=3
            shift
            ;;
        --conda|--option=4|--option4|-4)
            OPTION=4
            shift
            ;;
        *)
            log_error "未知选项: $1"
            SHOW_HELP=true
            shift
            ;;
    esac
done

# 显示帮助信息
show_help() {
    echo "用法: ./fix_pytorch.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --help, -h              显示此帮助信息"
    echo "  --auto                  自动选择最佳修复方案（基于环境检测）"
    echo "  --downgrade, -1         执行选项1: 降级依赖包"
    echo "  --cpu, -2               执行选项2: 安装CPU版本的PyTorch"
    echo "  --cuda, -3              执行选项3: 安装CUDA兼容版本的PyTorch"
    echo "  --conda, -4             执行选项4: 创建全新的Conda环境"
    echo ""
    echo "示例:"
    echo "  ./fix_pytorch.sh --auto           # 自动修复"
    echo "  ./fix_pytorch.sh --cpu            # 安装CPU版本"
    echo "  ./fix_pytorch.sh --cuda           # 安装CUDA兼容版本"
    echo ""
    exit 0
}

if [ "$SHOW_HELP" = true ]; then
    show_help
fi

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
    
    # 检查 NVIDIA 驱动版本和 CUDA 支持情况
    if command -v nvidia-smi &> /dev/null; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
        DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
        DRIVER_MINOR=$(echo $DRIVER_VERSION | cut -d. -f2)
        log_info "NVIDIA 驱动版本: $DRIVER_VERSION"
        
        # 根据驱动版本确定最大兼容的 CUDA 版本
        # 参考: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
        if [ $DRIVER_MAJOR -ge 525 ]; then
            MAX_CUDA="12.1"
        elif [ $DRIVER_MAJOR -ge 520 ]; then
            MAX_CUDA="12.0"
        elif [ $DRIVER_MAJOR -ge 510 ]; then
            MAX_CUDA="11.8"
        elif [ $DRIVER_MAJOR -ge 495 ]; then
            MAX_CUDA="11.5"
        elif [ $DRIVER_MAJOR -ge 470 ]; then
            MAX_CUDA="11.4"
        elif [ $DRIVER_MAJOR -ge 465 ]; then
            MAX_CUDA="11.3"
        elif [ $DRIVER_MAJOR -ge 460 ]; then
            MAX_CUDA="11.2"
        elif [ $DRIVER_MAJOR -ge 455 ]; then
            MAX_CUDA="11.1"
        elif [ $DRIVER_MAJOR -ge 450 ]; then
            MAX_CUDA="11.0"
        elif [ $DRIVER_MAJOR -ge 440 ]; then
            MAX_CUDA="10.2"
        elif [ $DRIVER_MAJOR -ge 418 ]; then
            MAX_CUDA="10.1"
        else
            MAX_CUDA="10.0"
        fi
        log_info "驱动支持的最大 CUDA 版本: $MAX_CUDA"
    else
        log_warning "未找到 NVIDIA 驱动，将使用 CPU 模式"
        MAX_CUDA="CPU"
    fi
    
    # 检查 PyTorch 版本
    if python -c "import torch" &> /dev/null; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        log_info "PyTorch 版本: $TORCH_VERSION"
        
        # 提取 PyTorch 的 CUDA 版本信息（如果有）
        if [[ $TORCH_VERSION == *+cu* ]]; then
            TORCH_CUDA_VERSION=$(echo $TORCH_VERSION | sed -n 's/.*+cu\([0-9]*\).*/\1/p')
            TORCH_CUDA_VERSION="${TORCH_CUDA_VERSION:0:2}.${TORCH_CUDA_VERSION:2}"
            log_info "PyTorch 需要的 CUDA 版本: $TORCH_CUDA_VERSION"
            
            # 检查 PyTorch CUDA 版本与驱动支持的最大 CUDA 版本是否兼容
            if [ "$MAX_CUDA" != "CPU" ]; then
                MAX_CUDA_MAJOR=$(echo $MAX_CUDA | cut -d. -f1)
                MAX_CUDA_MINOR=$(echo $MAX_CUDA | cut -d. -f2)
                TORCH_CUDA_MAJOR=$(echo $TORCH_CUDA_VERSION | cut -d. -f1)
                TORCH_CUDA_MINOR=$(echo $TORCH_CUDA_VERSION | cut -d. -f2)
                
                if [ $TORCH_CUDA_MAJOR -gt $MAX_CUDA_MAJOR ] || ([ $TORCH_CUDA_MAJOR -eq $MAX_CUDA_MAJOR ] && [ $TORCH_CUDA_MINOR -gt $MAX_CUDA_MINOR ]); then
                    log_warning "当前 PyTorch 需要 CUDA $TORCH_CUDA_VERSION，但您的驱动最高支持 CUDA $MAX_CUDA"
                    log_warning "这可能导致 CUDA 功能不可用或错误"
                    PYTORCH_CUDA_MISMATCH=true
                fi
            fi
        fi
        
        # 检查 CUDA 可用性
        CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
        if [ "$CUDA_AVAILABLE" = "True" ]; then
            CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
            log_info "CUDA 可用，版本: $CUDA_VERSION"
        else
            log_warning "CUDA 不可用"
            if [ "$MAX_CUDA" != "CPU" ]; then
                log_warning "虽然检测到 NVIDIA 驱动，但 PyTorch 无法使用 CUDA"
                log_warning "这可能是由于 PyTorch 和驱动版本不兼容导致的"
                PYTORCH_CUDA_MISMATCH=true
            fi
        fi
    else
        log_warning "未安装 PyTorch"
        TORCH_VERSION="未安装"
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
    
    # 显示警告和建议
    if [ "${PYTORCH_CUDA_MISMATCH}" = "true" ]; then
        echo ""
        log_warning "====================================================="
        log_warning "检测到 PyTorch 与 NVIDIA 驱动不兼容"
        log_warning "您的 PyTorch 版本 ($TORCH_VERSION) 需要更高版本的 CUDA"
        log_warning "而您的驱动 ($DRIVER_VERSION) 最高支持 CUDA $MAX_CUDA"
        log_warning "====================================================="
        log_warning "建议选择以下选项之一:"
        log_warning "1. 降级 PyTorch 至与驱动兼容的版本"
        log_warning "2. 使用 CPU 版本的 PyTorch"
        log_warning "3. 更新您的 NVIDIA 驱动"
        echo ""
    fi
    
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
        "12.1"|"12.0")
            log_info "安装 PyTorch 2.1.0 (CUDA 12.1)..."
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
            ;;
        "11.8"|"11.7"|"11.6")
            log_info "安装 PyTorch 2.0.1 (CUDA 11.7)..."
            pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
            ;;
        "11.5"|"11.4"|"11.3")
            log_info "安装 PyTorch 1.12.1 (CUDA 11.3)..."
            pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
            ;;
        "11.2"|"11.1")
            log_info "安装 PyTorch 1.10.1 (CUDA 11.1)..."
            pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 --extra-index-url https://download.pytorch.org/whl/cu111
            ;;
        "11.0")
            log_info "安装 PyTorch 1.7.1 (CUDA 11.0)..."
            pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
            ;;
        "10.2")
            log_info "安装 PyTorch 1.7.1 (CUDA 10.2)..."
            pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
            ;;
        "10.1")
            log_info "安装 PyTorch 1.5.1 (CUDA 10.1)..."
            pip install torch==1.5.1 torchvision==0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
            ;;
        "10.0")
            log_info "安装 PyTorch 1.2.0 (CUDA 10.0)..."
            pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
            ;;
        *)
            log_warning "无法确定兼容的 CUDA 版本，回退到 CPU 版本..."
            install_cpu_pytorch
            return
            ;;
    esac
    
    # 安装兼容的依赖包
    log_info "安装兼容的依赖包..."
    if [[ $MAX_CUDA == "12.1" || $MAX_CUDA == "12.0" || $MAX_CUDA == "11.8" || $MAX_CUDA == "11.7" || $MAX_CUDA == "11.6" ]]; then
        # 较新版本的 PyTorch (>=2.0)
        pip install transformers==4.34.0
        pip install datasets
        pip install bitsandbytes==0.41.1
        pip install accelerate==0.23.0
        pip install peft==0.6.0
    elif [[ $MAX_CUDA == "11.5" || $MAX_CUDA == "11.4" || $MAX_CUDA == "11.3" ]]; then
        # PyTorch 1.12.x
        pip install transformers==4.30.2
        pip install datasets
        pip install bitsandbytes-cuda113==0.27.2 || pip install bitsandbytes==0.27.2
        pip install accelerate==0.20.3
        pip install peft==0.4.0
    elif [[ $MAX_CUDA == "11.2" || $MAX_CUDA == "11.1" || $MAX_CUDA == "11.0" ]]; then
        # PyTorch 1.10.x 或 1.7.x
        pip install transformers==4.26.1
        pip install datasets
        pip install bitsandbytes==0.26.0
        pip install accelerate==0.16.0
        pip install peft==0.3.0
    else
        # 较旧版本的 PyTorch (<1.7)
        pip install transformers==4.20.1
        pip install datasets
        pip install accelerate==0.13.2
        # 旧版 PyTorch 可能不支持 bitsandbytes 和 peft
        log_warning "注意：您的 PyTorch 版本可能不支持最新的 bitsandbytes 和 peft 库"
        log_warning "可能需要手动调整代码以移除对这些库的依赖"
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
    conda create -n $ENV_NAME python=3.8 -y
    
    # 激活新环境
    log_info "激活环境 $ENV_NAME..."
    
    # 生成激活命令，让用户手动执行
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
            "12.1"|"12.0")
                echo "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121"
                echo "pip install transformers==4.34.0 datasets bitsandbytes==0.41.1 accelerate==0.23.0 peft==0.6.0 wandb"
                ;;
            "11.8"|"11.7"|"11.6")
                echo "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117"
                echo "pip install transformers==4.34.0 datasets bitsandbytes==0.41.1 accelerate==0.23.0 peft==0.6.0 wandb"
                ;;
            "11.5"|"11.4"|"11.3")
                echo "pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113"
                echo "pip install transformers==4.30.2 datasets bitsandbytes-cuda113==0.27.2 accelerate==0.20.3 peft==0.4.0 wandb"
                ;;
            "11.2"|"11.1")
                echo "pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 --extra-index-url https://download.pytorch.org/whl/cu111"
                echo "pip install transformers==4.26.1 datasets bitsandbytes==0.26.0 accelerate==0.16.0 peft==0.3.0 wandb"
                ;;
            "11.0")
                echo "pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html"
                echo "pip install transformers==4.26.1 datasets accelerate==0.16.0 peft==0.3.0 wandb"
                ;;
            "10.2")
                echo "pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html"
                echo "pip install transformers==4.26.1 datasets accelerate==0.16.0 peft==0.3.0 wandb"
                ;;
            "10.1"|"10.0")
                echo "pip install torch==1.5.1 torchvision==0.6.1 -f https://download.pytorch.org/whl/torch_stable.html"
                echo "pip install transformers==4.20.1 datasets accelerate==0.13.2 wandb"
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
    
    # 如果启用自动修复，根据环境选择最佳选项
    if [ "$AUTO_FIX" = true ]; then
        log_info "自动选择最佳修复方案..."
        
        if [ "$MAX_CUDA" = "CPU" ]; then
            # 没有 NVIDIA 驱动，选择 CPU 模式
            OPTION=2
            log_info "自动选择: 安装 CPU 版本的 PyTorch"
        elif [ "${PYTORCH_CUDA_MISMATCH}" = "true" ]; then
            # PyTorch 与驱动不兼容，安装兼容的 CUDA 版本
            OPTION=3
            log_info "自动选择: 安装 CUDA 兼容版本的 PyTorch"
        elif [ -z "$TORCH_VERSION" ] || [ "$TORCH_VERSION" = "未安装" ]; then
            # 未安装 PyTorch，安装兼容的 CUDA 版本
            OPTION=3
            log_info "自动选择: 安装 CUDA 兼容版本的 PyTorch"
        else
            # 已有可工作的 PyTorch，只降级依赖包
            OPTION=1
            log_info "自动选择: 降级依赖包"
        fi
    fi
    
    # 如果未通过命令行或自动选择指定选项，则显示交互式菜单
    if [ -z "$OPTION" ]; then
        show_options
    fi
    
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