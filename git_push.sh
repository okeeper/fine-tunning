#!/bin/bash

# 快速Git提交和推送脚本
# 用法: ./git_push.sh "提交信息" [分支名称]

# 显示帮助信息
show_help() {
    echo "用法: ./git_push.sh \"提交信息\" [分支名称]"
    echo ""
    echo "参数:"
    echo "  提交信息    必需参数，Git提交的描述信息"
    echo "  分支名称    可选参数，要推送的分支名称，默认为当前分支"
    echo ""
    echo "示例:"
    echo "  ./git_push.sh \"更新README文档\""
    echo "  ./git_push.sh \"修复bug\" dev"
    echo ""
}

# 检查是否是Git仓库
if [ ! -d ".git" ]; then
    echo "错误: 当前目录不是Git仓库"
    echo "请在Git仓库根目录下运行此脚本"
    exit 1
fi

# 检查提交信息是否提供
if [ -z "$1" ]; then
    echo "错误: 未提供提交信息"
    show_help
    exit 1
fi

# 获取提交信息
COMMIT_MESSAGE="$1"

# 获取当前分支
CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "错误: 无法获取当前分支"
    exit 1
fi

# 设置要推送的分支
BRANCH_TO_PUSH=${2:-$CURRENT_BRANCH}

# 显示状态信息
echo "=================================================="
echo "  Git快速提交和推送"
echo "=================================================="
echo "当前分支: $CURRENT_BRANCH"
echo "推送分支: $BRANCH_TO_PUSH"
echo "提交信息: $COMMIT_MESSAGE"
echo ""

# 显示变更文件
echo "变更文件:"
git status -s
echo ""

# 添加所有变更
echo "添加所有变更..."
git add .

# 提交变更
echo "提交变更..."
git commit -m "$COMMIT_MESSAGE"

# 如果指定的分支不是当前分支，则切换分支
if [ "$BRANCH_TO_PUSH" != "$CURRENT_BRANCH" ]; then
    echo "切换到分支 $BRANCH_TO_PUSH..."
    
    # 检查分支是否存在
    git show-ref --verify --quiet refs/heads/$BRANCH_TO_PUSH
    if [ $? -ne 0 ]; then
        # 分支不存在，创建新分支
        echo "分支 $BRANCH_TO_PUSH 不存在，创建新分支..."
        git checkout -b $BRANCH_TO_PUSH
    else
        # 分支存在，切换到该分支
        git checkout $BRANCH_TO_PUSH
    fi
fi

# 推送到远程仓库
echo "推送到远程仓库..."
git push origin $BRANCH_TO_PUSH

# 如果切换了分支，切回原来的分支
if [ "$BRANCH_TO_PUSH" != "$CURRENT_BRANCH" ]; then
    echo "切回到原分支 $CURRENT_BRANCH..."
    git checkout $CURRENT_BRANCH
fi

echo ""
echo "=================================================="
echo "  操作完成!"
echo "=================================================="
echo "提交信息: $COMMIT_MESSAGE"
echo "推送分支: $BRANCH_TO_PUSH"
echo "" 