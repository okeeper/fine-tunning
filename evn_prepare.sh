# 检查并创建conda环境
ENV_NAME="llama_ft_env"

# 检查conda是否已安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 检查环境是否已存在
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "创建新的conda环境: $ENV_NAME..."
    conda create -n $ENV_NAME python=3.10 -y
    echo "conda环境创建成功"
else
    echo "conda环境 $ENV_NAME 已存在"
fi

# 激活环境
echo "激活conda环境: $ENV_NAME..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 检查环境是否成功激活
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "错误: 无法激活conda环境 $ENV_NAME"
    exit 1
fi

echo "成功激活conda环境: $ENV_NAME"

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt