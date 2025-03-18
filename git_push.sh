#!/bin/bash

# 优化版Git提交和推送脚本
# 支持自动添加未提交文件、自定义提交信息和交互式确认

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # 重置颜色

# 显示帮助信息
function show_help {
    echo -e "${BLUE}优化版Git提交和推送工具${NC}"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --message MSG     提交信息 (如果不提供则会提示输入)"
    echo "  -a, --all             添加所有改动，包括未跟踪的文件"
    echo "  -b, --branch BRANCH   指定要推送的分支 (默认: 当前分支)"
    echo "  -r, --remote REMOTE   指定远程仓库名称 (默认: origin)"
    echo "  -s, --skip-push       只提交不推送"
    echo "  -y, --yes             跳过确认提示"
    echo "  -h, --help            显示此帮助信息"
    echo ""
}

# 显示错误并退出
function error_exit {
    echo -e "${RED}错误: $1${NC}" >&2
    exit 1
}

# 显示信息
function log_info {
    echo -e "${GREEN}[INFO] $1${NC}"
}

# 显示警告
function log_warning {
    echo -e "${YELLOW}[警告] $1${NC}"
}

# 检查是否在git仓库中
function check_git_repo {
    if ! git rev-parse --is-inside-work-tree &>/dev/null; then
        error_exit "当前目录不是Git仓库"
    fi
}

# 获取当前分支
function get_current_branch {
    git branch --show-current 2>/dev/null || 
    git rev-parse --abbrev-ref HEAD 2>/dev/null || 
    echo "main"
}

# 输出彩色的git状态
function print_git_status {
    echo -e "${CYAN}Git状态:${NC}"
    git -c color.status=always status
}

# 解析参数
COMMIT_MSG=""
ADD_ALL=0
BRANCH=""
REMOTE="origin"
SKIP_PUSH=0
SKIP_CONFIRM=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--message)
            COMMIT_MSG="$2"
            shift 2
            ;;
        -a|--all)
            ADD_ALL=1
            shift
            ;;
        -b|--branch)
            BRANCH="$2"
            shift 2
            ;;
        -r|--remote)
            REMOTE="$2"
            shift 2
            ;;
        -s|--skip-push)
            SKIP_PUSH=1
            shift
            ;;
        -y|--yes)
            SKIP_CONFIRM=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            # 如果没有指定-m但提供了参数，假设是提交信息
            if [[ -z "$COMMIT_MSG" && "$1" != -* ]]; then
                COMMIT_MSG="$1"
                shift
            else
                error_exit "未知选项: $1"
            fi
            ;;
    esac
done

# 检查是否在git仓库中
check_git_repo

# 如果未指定分支，使用当前分支
if [[ -z "$BRANCH" ]]; then
    BRANCH=$(get_current_branch)
    log_info "使用当前分支: $BRANCH"
fi

# 显示当前状态
print_git_status

# 检查未提交的文件
UNTRACKED_FILES=$(git ls-files --others --exclude-standard | wc -l)
MODIFIED_FILES=$(git ls-files --modified | wc -l)
DELETED_FILES=$(git ls-files --deleted | wc -l)
STAGED_FILES=$(git diff --staged --name-only | wc -l)

# 统计文件数
CHANGES_TO_COMMIT=$((UNTRACKED_FILES + MODIFIED_FILES + DELETED_FILES))

if [[ $CHANGES_TO_COMMIT -eq 0 && $STAGED_FILES -eq 0 ]]; then
    log_warning "没有要提交的更改"
    exit 0
fi

# 自动添加文件
if [[ $ADD_ALL -eq 1 ]]; then
    log_info "添加所有更改的文件..."
    git add -A
else
    # 提示用户添加未跟踪的文件
    if [[ $UNTRACKED_FILES -gt 0 ]]; then
        echo -e "${YELLOW}发现 $UNTRACKED_FILES 个未跟踪的文件${NC}"
        
        if [[ $SKIP_CONFIRM -eq 0 ]]; then
            read -p "是否添加所有未跟踪的文件? (y/n): " ADD_UNTRACKED
            if [[ "$ADD_UNTRACKED" =~ ^[Yy]$ ]]; then
                git ls-files --others --exclude-standard | xargs git add
                log_info "已添加未跟踪的文件"
            fi
        else
            # 自动添加未跟踪文件（如果指定了-y选项）
            git ls-files --others --exclude-standard | xargs git add
            log_info "已自动添加未跟踪的文件"
        fi
    fi
    
    # 提示用户添加已修改的文件
    if [[ $MODIFIED_FILES -gt 0 || $DELETED_FILES -gt 0 ]]; then
        echo -e "${YELLOW}发现 $MODIFIED_FILES 个已修改的文件和 $DELETED_FILES 个已删除的文件${NC}"
        
        if [[ $SKIP_CONFIRM -eq 0 ]]; then
            read -p "是否添加所有已修改/删除的文件? (y/n): " ADD_MODIFIED
            if [[ "$ADD_MODIFIED" =~ ^[Yy]$ ]]; then
                git add -u
                log_info "已添加所有已修改/删除的文件"
            fi
        else
            # 自动添加已修改/删除的文件（如果指定了-y选项）
            git add -u
            log_info "已自动添加所有已修改/删除的文件"
        fi
    fi
fi

# 再次显示状态
print_git_status

# 检查是否有暂存的更改
STAGED_FILES=$(git diff --staged --name-only | wc -l)
if [[ $STAGED_FILES -eq 0 ]]; then
    log_warning "没有暂存的更改可提交"
    exit 0
fi

# 提示输入提交信息（如果未提供）
if [[ -z "$COMMIT_MSG" ]]; then
    echo -e "${CYAN}请输入提交信息:${NC}"
    read -p "> " COMMIT_MSG
    
    if [[ -z "$COMMIT_MSG" ]]; then
        COMMIT_MSG="更新代码 $(date +'%Y-%m-%d %H:%M:%S')"
        log_warning "使用默认提交信息: $COMMIT_MSG"
    fi
fi

# 提交更改
log_info "提交更改..."
git commit -m "$COMMIT_MSG"

# 如果没有跳过推送，则推送到远程
if [[ $SKIP_PUSH -eq 0 ]]; then
    if [[ $SKIP_CONFIRM -eq 0 ]]; then
        read -p "是否推送到远程 $REMOTE/$BRANCH? (y/n): " DO_PUSH
        if [[ "$DO_PUSH" =~ ^[Yy]$ ]]; then
            log_info "推送到远程 $REMOTE/$BRANCH..."
            git push $REMOTE $BRANCH
        else
            log_info "跳过推送，只进行了本地提交"
        fi
    else
        # 如果指定了-y选项，自动推送
        log_info "推送到远程 $REMOTE/$BRANCH..."
        git push $REMOTE $BRANCH
    fi
else
    log_info "跳过推送，只进行了本地提交"
fi

log_info "操作完成！"

# 显示最近的提交
echo -e "${CYAN}最近提交:${NC}"
git log -1 --stat 