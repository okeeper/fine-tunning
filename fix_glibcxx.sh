#!/bin/bash

# 修复GLIBCXX_3.4.32 not found错误的脚本

echo "=================================================="
echo "  修复GLIBCXX_3.4.32 not found错误"
echo "=================================================="
echo ""

# 检查是否在conda环境中
if [ -z "$CONDA_PREFIX" ]; then
    echo "错误: 未检测到激活的conda环境"
    echo "请先激活conda环境，例如: conda activate llama_ft_env"
    exit 1
fi

echo "当前conda环境: $CONDA_PREFIX"
echo ""

# 备份原始库文件
echo "步骤1: 备份原始库文件..."
if [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
    echo "备份 $CONDA_PREFIX/lib/libstdc++.so.6 到 $CONDA_PREFIX/lib/libstdc++.so.6.backup"
    cp "$CONDA_PREFIX/lib/libstdc++.so.6" "$CONDA_PREFIX/lib/libstdc++.so.6.backup"
fi
echo "备份完成"
echo ""

# 安装更新的库
echo "步骤2: 安装更新的库..."
echo "运行: conda install -c conda-forge libstdcxx-ng -y"
conda install -c conda-forge libstdcxx-ng -y

# 检查安装是否成功
if [ $? -ne 0 ]; then
    echo "错误: 库安装失败"
    echo "请尝试手动运行: conda install -c conda-forge libstdcxx-ng -y"
    exit 1
fi
echo "库安装完成"
echo ""

# 验证修复
echo "步骤3: 验证修复..."
echo "检查libstdc++.so.6是否包含GLIBCXX_3.4.32..."

if [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
    GLIBCXX_CHECK=$(strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep GLIBCXX_3.4.32)
    if [ -n "$GLIBCXX_CHECK" ]; then
        echo "成功: 找到GLIBCXX_3.4.32"
        echo "修复完成！"
    else
        echo "警告: 未找到GLIBCXX_3.4.32"
        echo "修复可能不完整，请尝试其他方法："
        echo "1. 使用CPU模式运行: bash run_finetune.sh --use_cpu"
        echo "2. 在Docker容器中运行"
    fi
else
    echo "警告: 未找到libstdc++.so.6文件"
    echo "修复可能不完整，请尝试其他方法"
fi

echo ""
echo "=================================================="
echo "  修复过程完成"
echo "=================================================="
echo ""
echo "如果问题仍然存在，您可以尝试以下方法："
echo "1. 使用CPU模式运行: bash run_finetune.sh --use_cpu"
echo "2. 使用Docker容器运行"
echo "3. 更新系统CUDA驱动程序"
echo "" 