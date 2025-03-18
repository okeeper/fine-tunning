#!/bin/bash

# 先使自身可执行
chmod +x $(readlink -f "$0")

echo "=================================================="
echo "  Git快速提交和推送"
echo "=================================================="

# 获取当前分支名称
BRANCH=$(git branch --show-current)

# 设置默认推送分支
if [ -z "$BRANCH" ]; then
    BRANCH="main"
fi

echo "当前分支: $BRANCH"
echo "推送分支: $BRANCH"
echo "提交信息: $1"
echo ""

# 检查是否提供了提交信息
if [ -z "$1" ]; then
    echo "错误: 未提供提交信息"
    echo "用法: ./git_push.sh \"提交信息\""
    exit 1
fi

# 显示变更文件
echo "变更文件:"
git status --porcelain

# 为所有脚本添加执行权限
echo ""
echo "为脚本添加执行权限..."
chmod +x *.sh
chmod +x src/*.py
chmod +x data/*.py 2>/dev/null || true

# 添加所有变更
echo "添加所有变更..."
git add .

# 提交变更
echo "提交变更..."
git commit -m "$1"

# 推送到远程仓库
echo "推送到远程仓库..."
git push origin $BRANCH

echo ""
echo "=================================================="
echo "  操作完成!"
echo "=================================================="
echo "提交信息: $1"
echo "推送分支: $BRANCH" 