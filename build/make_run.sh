#!/bin/bash

# 设置脚本遇到错误立即退出，避免错误累积
set -e

# 定义颜色输出，让用户更容易识别不同级别的信息
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 工具基本信息配置
TOOL_NAME="msmemscope"
RUN_FILE="${TOOL_NAME}.run"
BUILD_DIR="$(cd "$(dirname "$0")" && pwd)"  # 获取脚本所在绝对路径
TEMP_DIR="/tmp/${TOOL_NAME}_build_$$"      # 使用进程ID确保临时目录唯一性

# 源目录定义 - 这些是需要打包的交付件目录
PYTHON_DIR="../python"
BIN_DIR="../output/bin"
LIB64_DIR="../output/lib64"

# 日志输出函数 - 所有用户可见的输出都是英文
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查源目录是否存在
check_source_dirs() {
    local missing_dirs=()
    
    # 检查每个必需的目录是否存在
    [ ! -d "$PYTHON_DIR" ] && missing_dirs+=("$PYTHON_DIR")
    [ ! -d "$BIN_DIR" ] && missing_dirs+=("$BIN_DIR")
    [ ! -d "$LIB64_DIR" ] && missing_dirs+=("$LIB64_DIR")
    
    # 如果有目录缺失，报告错误
    if [ ${#missing_dirs[@]} -gt 0 ]; then
        log_error "The following source directories are missing:"
        for dir in "${missing_dirs[@]}"; do
            echo "  $dir"
        done
        return 1
    fi
    
    return 0
}

# 创建临时构建目录
create_temp_dir() {
    mkdir -p "$TEMP_DIR"
    mkdir -p "$TEMP_DIR/payload"  # payload目录用于存放要打包的文件
}

# 复制交付件到临时目录
copy_artifacts() {
    log_info "Copying artifacts to temporary directory..."
    
    # 在payload下创建msmemscope目录
    mkdir -p "$TEMP_DIR/payload/msmemscope"
    
    # 使用rsync保持文件权限和属性，如果rsync不可用则用cp
    if command -v rsync >/dev/null 2>&1; then
        rsync -av "$PYTHON_DIR/" "$TEMP_DIR/payload/msmemscope/python/" --exclude="*.pyc" --exclude="__pycache__"
        rsync -av "$BIN_DIR/" "$TEMP_DIR/payload/msmemscope/bin/"
        rsync -av "$LIB64_DIR/" "$TEMP_DIR/payload/msmemscope/lib64/"
    else
        mkdir -p "$TEMP_DIR/payload/msmemscope"
        cp -r "$PYTHON_DIR" "$TEMP_DIR/payload/msmemscope/"
        cp -r "$BIN_DIR" "$TEMP_DIR/payload/msmemscope/"
        cp -r "$LIB64_DIR" "$TEMP_DIR/payload/msmemscope/"
    fi
    
    # 创建版本信息文件，放在msmemscope目录下
    echo "version: 1.0.0" > "$TEMP_DIR/payload/msmemscope/version.txt"
    echo "build_date: $(date '+%Y-%m-%d %H:%M:%S')" >> "$TEMP_DIR/payload/msmemscope/version.txt"
    echo "build_hash: $(date +%s | sha256sum | head -c 8)" >> "$TEMP_DIR/payload/msmemscope/version.txt"
}

# 创建安装脚本 - 这个脚本会被嵌入到run文件中
create_install_script() {
    cat > "$TEMP_DIR/install.sh" << 'EOF'
#!/bin/bash

# 设置严格模式，遇到错误立即退出
set -e

# 颜色定义用于用户输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 安装配置
TOOL_NAME="msmemscope"
DEFAULT_INSTALL_PATH="."
BACKUP_DIR="/tmp/${TOOL_NAME}_backup_$$"  # 使用进程ID确保备份目录唯一

# 日志函数 - 所有用户输出都是英文
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查磁盘空间是否足够
check_disk_space() {
    local install_path="$1"
    local required_space=100  # 最小需要100MB空间
    
    # 获取安装路径所在分区的可用空间（MB）
    local available_space=$(df -m "$install_path" | awk 'NR==2 {print $4}')
    
    if [ "$available_space" -lt "$required_space" ]; then
        log_error "Insufficient disk space! Required: ${required_space}MB, Available: ${available_space}MB"
        return 1
    fi
    
    log_info "Disk space check passed: ${available_space}MB available"
    return 0
}

# 验证安装路径的合法性
validate_install_path() {
    local install_path="$1"
    
    # 检查路径是否为空
    if [ -z "$install_path" ]; then
        log_error "Install path cannot be empty"
        return 1
    fi
    
    # 检查是否为绝对路径
    if [[ "$install_path" != /* ]]; then
        log_error "Install path must be an absolute path: $install_path"
        return 1
    fi

    # 检查安装目录本身是否存在 - 新增的关键检查
    if [ -e "$install_path" ]; then
        if [ ! -d "$install_path" ]; then
            log_error "Install path exists but is not a directory: $install_path"
            return 1
        fi
    else
        log_error "Install directory does not exist: $install_path"
        return 1
    fi
    
    # 检查安装目录是否有写入权限
    if [ ! -w "$install_path" ]; then
        log_error "No write permission for directory: $install_path"
        return 1
    fi
    
    # 检查路径是否包含特殊字符
    if [[ "$install_path" =~ [\|\<\>\"\'] ]]; then
        log_error "Install path contains invalid characters"
        return 1
    fi
    
    return 0
}

# 检查是否已经安装了该工具
check_installed() {
    local install_path="$1"
    
    log_info "check_installed: Checking directory: $install_path/msmemscope"  # 内部调试
    
    if [ -d "$install_path/msmemscope" ]; then
        log_info "check_installed: Directory exists"  # 内部调试
        
        # 安全地检查目录是否为空，避免set -e导致退出
        local is_empty=true
        if ls -A "$install_path/msmemscope" >/dev/null 2>&1; then
            # ls命令成功，说明目录不为空
            log_info "check_installed: Directory is NOT empty"  # 内部调试
            is_empty=false
        else
            log_info "check_installed: Directory is empty"  # 内部调试
        fi
        
        if [ "$is_empty" = true ]; then
            log_info "Target directory is empty, proceeding with fresh installation"
            return 2  # 空目录
        elif [ -f "$install_path/msmemscope/version.txt" ]; then
            # 有效安装
            log_info "check_installed: Valid installation found"  # 内部调试
            return 0
        else
            # 目录非空但没有version.txt，可能是其他软件
            log_warn "Directory exists but may not be a valid installation: $install_path/msmemscope"
            log_info "check_installed: Returning status 1"  # 内部调试
            return 1
        fi
    else
        # 目录不存在
        log_info "Target directory does not exist, will create during installation"
        log_info "check_installed: Returning status 3"  # 内部调试
        return 3  # 目录不存在
    fi
    
    log_info "check_installed: Falling through to default return"  # 内部调试
    return 2
}

# 备份现有安装（用于升级或重新安装场景）
backup_existing() {
    local install_path="$1"
    
    if [ -d "$install_path/msmemscope" ]; then
        mkdir -p "$BACKUP_DIR"
        log_info "Backing up existing installation to: $BACKUP_DIR"
        # 使用cp -r备份msmemscope目录，忽略可能的权限错误
        cp -r "$install_path/msmemscope" "$BACKUP_DIR/" 2>/dev/null || true
    fi
}

# 恢复备份（在安装失败时使用）
restore_backup() {
    local install_path="$1"
    
    if [ -d "$BACKUP_DIR" ] && [ -d "$BACKUP_DIR/msmemscope" ]; then
        log_info "Restoring from backup..."
        rm -rf "$install_path/msmemscope"
        mv "$BACKUP_DIR/msmemscope" "$install_path/"
        rm -rf "$BACKUP_DIR"
        log_info "Backup restored successfully"
    fi
}

# 执行实际的安装操作
perform_installation() {
    local install_path="$1"
    local is_upgrade="$2"
    
    local action="${is_upgrade:-Installing}"
    log_info "${action} to: $install_path/msmemscope"
    
    # 创建安装目录，-p参数确保父目录也存在
    mkdir -p "$install_path"
    
    # 从run文件中提取payload部分并解压
    log_info "Extracting files..."
    # 找到payload起始行号
    PAYLOAD_START=$(awk '/^__PAYLOAD_BELOW__/ {print NR + 1; exit 0; }' "$0")
    # 从payload起始行开始提取并解压到安装目录
    tail -n +$PAYLOAD_START "$0" | tar -xz -C "$install_path"
    
    # 设置文件权限，确保可执行文件有执行权限
    if [ -d "$install_path/msmemscope/bin" ]; then
        chmod -R 755 "$install_path/msmemscope/bin"
        log_info "Set execute permissions for bin directory"
    fi
    
    if [ -d "$install_path/msmemscope/python" ]; then
        chmod -R 755 "$install_path/msmemscope/python"
        log_info "Set execute permissions for python directory"
    fi
    
    log_info "File ${is_upgrade:-installation} completed"
}

# 创建卸载脚本
create_uninstall_script() {
    local install_path="$1"
    
    # 生成卸载脚本文件，放在msmemscope目录下
    cat > "$install_path/msmemscope/uninstall.sh" << 'UNINSTALL_EOF'
#!/bin/bash

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取安装目录（卸载脚本所在目录）
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"

# 检查安装完整性，确保是有效的安装
check_installation_integrity() {
    # 检查版本文件是否存在
    if [ ! -f "$INSTALL_DIR/version.txt" ]; then
        log_error "Installation directory is incomplete or corrupted"
        return 1
    fi
    
    # 检查必要的目录是否存在
    local required_dirs=("bin" "python" "lib64")
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$INSTALL_DIR/$dir" ]; then
            log_error "Missing required directory: $dir"
            return 1
        fi
    done
    
    return 0
}

# 确认卸载操作，避免误操作
confirm_uninstall() {
    echo "=============================================="
    echo "           Uninstall msmemscope"
    echo "=============================================="
    echo "Installation directory: $INSTALL_DIR"
    echo "Version: $(grep 'version:' $INSTALL_DIR/version.txt 2>/dev/null | cut -d' ' -f2 || echo 'Unknown')"
    echo ""
    
    # 显示警告信息
    echo -e "${YELLOW}Warning: This operation will permanently delete ALL files and subdirectories${NC}"
    echo -e "${YELLOW}within the installation directory, including any user-created content!${NC}"
    echo ""
    echo -e "${RED}The following directory and ALL its contents will be deleted:${NC}"
    echo -e "${RED}  $INSTALL_DIR${NC}"
    echo ""
    
    read -p "Are you sure you want to uninstall? (y/N): " confirm
    case "$confirm" in
        [yY]|[yY][eE][sS])
            return 0
            ;;
        *)
            echo "Uninstall cancelled"
            exit 0
            ;;
    esac
}

# 检查是否有进程正在使用安装目录
check_running_processes() {
    log_info "Checking for processes using the installation directory..."
    
    # 使用lsof检查是否有进程在使用安装目录
    if command -v lsof >/dev/null 2>&1; then
        if lsof +D "$INSTALL_DIR" 2>/dev/null | grep -q "."; then
            log_warn "Found processes using the installation directory:"
            lsof +D "$INSTALL_DIR" 2>/dev/null | head -10
            return 1
        fi
    fi
    
    # 使用fuser作为备选检查方法
    if command -v fuser >/dev/null 2>&1; then
        if fuser "$INSTALL_DIR" 2>/dev/null; then
            log_warn "Found processes using the installation directory"
            return 1
        fi
    fi
    
    return 0
}

# 执行卸载操作
perform_uninstall() {
    log_info "Starting uninstallation..."
    
    # 检查是否有进程在使用
    if ! check_running_processes; then
        log_warn "Processes are using the installation directory"
        read -p "Force uninstall anyway? (y/N): " force_uninstall
        case "$force_uninstall" in
            [yY]|[yY][eE][sS])
                log_warn "Forcing uninstall..."
                ;;
            *)
                echo "Uninstall cancelled"
                exit 1
                ;;
        esac
    fi
    
    # 删除安装目录
    log_info "Removing installation directory: $INSTALL_DIR"
    rm -rf "$INSTALL_DIR"
    
    log_info "Uninstallation completed successfully"
}

# 主函数
main() {
    echo "=============================================="
    echo "          msmemscope Uninstaller"
    echo "=============================================="
    
    # 检查安装完整性
    if ! check_installation_integrity; then
        log_error "Installation directory is incomplete, manual cleanup may be required"
        exit 1
    fi
    
    # 确认卸载
    confirm_uninstall
    
    # 执行卸载
    perform_uninstall
}

# 脚本入口点
main "$@"
UNINSTALL_EOF

    # 设置卸载脚本为可执行
    chmod +x "$install_path/msmemscope/uninstall.sh"
    log_info "Uninstall script created: $install_path/msmemscope/uninstall.sh"
}

# 显示安装完成信息
show_installation_info() {
    local install_path="$1"
    local is_upgrade="$2"
    
    # 规范化路径，移除多余的 ./ 和 ../
    local normalized_path=$(cd "$install_path" && pwd)
    
    local action="${is_upgrade:-Installation}"
    echo ""
    echo "=============================================="
    echo "          $TOOL_NAME ${action} Complete"
    echo "=============================================="
    echo "Installation path: $normalized_path/msmemscope"
    echo "Version: $(grep 'version:' "$install_path/msmemscope/version.txt" | cut -d' ' -f2)"
    echo "Install time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 如果是升级操作，显示升级成功信息
    if [ -n "$is_upgrade" ]; then
        log_info "Upgrade completed successfully"
    else
        log_info "Installation completed successfully"
    fi
}

# 安装模式的主函数
install_main() {
    local install_path=""
    local is_upgrade=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --install-path=*)
                install_path="${1#*=}"
                shift
                ;;
            --upgrade)
                is_upgrade=true
                shift
                ;;
            *)
                log_warn "Unknown parameter: $1"
                shift
                ;;
        esac
    done
    
    log_info "Starting installation to: $install_path"
    
    # 设置默认安装路径时，如果是相对路径就转换为绝对路径
    if [ -z "$install_path" ]; then
        install_path="$DEFAULT_INSTALL_PATH"
        # 将相对路径转换为绝对路径
        if [[ "$install_path" != /* ]]; then
            install_path="$(pwd)/$install_path"
        fi
        log_info "Using installation path: $install_path"
    fi
    
    # 验证安装路径
    validate_install_path "$install_path" || exit 1
    
    # 检查磁盘空间
    check_disk_space "$install_path" || exit 1
    
    # 检查安装状态
    if [ -d "$install_path/msmemscope" ] && [ -f "$install_path/msmemscope/version.txt" ]; then
        if [ "$is_upgrade" = true ]; then
            # 升级模式：需要备份
            log_info "Starting upgrade process..."
            backup_existing "$install_path"
        else
            # 普通安装模式：已存在就报错退出
            log_warn "Tool is already installed. Use upgrade mode or uninstall first."
            exit 1
        fi
    elif [ "$is_upgrade" = true ]; then
        # 升级模式但目录无效
        log_error "Target directory is not a valid installation for upgrade"
        exit 1
    else
        # 全新安装
        log_info "Starting fresh installation..."
    fi
    
    # 执行安装操作
    perform_installation "$install_path" "$([ "$is_upgrade" = true ] && echo "Upgrading")"
    create_uninstall_script "$install_path"
    show_installation_info "$install_path" "$([ "$is_upgrade" = true ] && echo "Upgrade")"
}

# 升级模式的主函数
upgrade_main() {
    local install_path=""
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --install-path=*)
                install_path="${1#*=}"
                shift
                ;;
            *)
                log_warn "Unknown parameter: $1"
                shift
                ;;
        esac
    done
    
    # 检查必须参数
    if [ -z "$install_path" ]; then
        log_error "Upgrade requires --install-path parameter"
        echo "Usage: $0 --upgrade --install-path=<path>"
        exit 1
    fi
    
    # 检查目标目录是否存在
    if [ ! -d "$install_path" ]; then
        log_error "Target installation directory does not exist: $install_path"
        exit 1
    fi
    
    # 验证目标目录是否是有效的安装
    if [ ! -f "$install_path/msmemscope/version.txt" ]; then
        log_error "Target directory is not a valid $TOOL_NAME installation"
        exit 1
    fi
    
    log_info "Starting upgrade for: $install_path"
    # 调用安装主函数，并设置升级标志
    install_main --install-path="$install_path" --upgrade
}

# 显示版本信息
show_version() {
    echo "=============================================="
    echo "           $TOOL_NAME Version Info"
    echo "=============================================="
    
    # 检查是否在run文件内部（安装前）
    if [ -f "version.txt" ]; then
        # 在构建目录中
        cat "version.txt"
    else
        # 尝试从run文件的payload中提取版本信息
        local payload_start=$(awk '/^__PAYLOAD_BELOW__/ {print NR + 1; exit 0; }' "$0")
        if [ -n "$payload_start" ]; then
            # 提取version.txt文件内容
            tail -n +$payload_start "$0" | tar -xz -O "msmemscope/version.txt" 2>/dev/null || \
            echo "Version: 1.0.0 (build $(date '+%Y%m%d'))"
        else
            echo "Version: 1.0.0 (build $(date '+%Y%m%d'))"
        fi
    fi
    echo ""
}

# 显示帮助信息
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --install               Install the tool"
    echo "  --install-path=PATH     Specify installation path (must be absolute)"
    echo "  --upgrade               Upgrade an existing installation"
    echo "  --version               Show version information"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --install                                    # Install to default path"
    echo "  $0 --install --install-path=/usr/local/msmemscope"
    echo "  $0 --upgrade --install-path=/opt/msmemscope"
    echo "  $0 --version                                    # Show version"
    echo "  $0 --help"
    echo ""
}

# 主执行逻辑 - 根据参数调用不同的功能模块
main() {
    # 检查是否有参数，没有则显示帮助
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    case "$1" in
        --install)
            shift
            install_main "$@"
            ;;
        --upgrade)
            shift
            upgrade_main "$@"
            ;;
        --version|-v)
            show_version
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# 脚本入口点
main "$@"
exit 0

# 标记payload开始的位置，安装脚本会从这里开始读取压缩包
__PAYLOAD_BELOW__
EOF
}

# 创建最终的run文件
create_run_file() {
    log_info "Creating run file: $RUN_FILE"
    
    # 复制安装脚本到run文件
    cp "$TEMP_DIR/install.sh" "$BUILD_DIR/$RUN_FILE"
    
    # 打包payload并追加到run文件
    log_info "Packaging payload..."
    cd "$TEMP_DIR/payload"
    tar -cz . >> "$BUILD_DIR/$RUN_FILE"  # 使用.而不是*避免隐藏文件被忽略
    cd - > /dev/null
    
    # 设置run文件为可执行
    chmod +x "$BUILD_DIR/$RUN_FILE"
    
    # 验证run文件是否创建成功
    if [ -f "$BUILD_DIR/$RUN_FILE" ] && [ -x "$BUILD_DIR/$RUN_FILE" ]; then
        log_info "Run file created successfully: $BUILD_DIR/$RUN_FILE"
    else
        log_error "Failed to create run file"
        return 1
    fi
}

# 清理临时文件
cleanup() {
    if [ -d "$TEMP_DIR" ]; then
        log_info "Cleaning up temporary files..."
        rm -rf "$TEMP_DIR"
    fi
}

# 显示构建完成信息
show_build_info() {
    local run_file_size=$(du -h "$RUN_FILE" | cut -f1)
    local run_file_path="$BUILD_DIR/$RUN_FILE"
    
    # 从version.txt获取版本信息
    local version_info=""
    if [ -f "$TEMP_DIR/payload/msmemscope/version.txt" ]; then
        version_info=$(grep "version:" "$TEMP_DIR/payload/msmemscope/version.txt" | cut -d' ' -f2)
    fi
    
    echo ""
    echo "=============================================="
    echo "           $TOOL_NAME Run Package Built"
    echo "=============================================="
    echo "File: $RUN_FILE"
    echo "Size: $run_file_size"
    echo "Location: $run_file_path"
    echo "Version: ${version_info:-1.0.0}"
    echo ""
    echo "Usage instructions:"
    echo "  Install: bash $RUN_FILE --install [--install-path=/path]"
    echo "  Upgrade: bash $RUN_FILE --upgrade --install-path=/path"
    echo "  Version: bash $RUN_FILE --version"
    echo "  Help:    bash $RUN_FILE --help"
    echo ""
    log_info "Build process completed successfully"
}

# 主构建函数
main() {
    log_info "Starting build process for $RUN_FILE ..."
    
    # 检查源目录
    if ! check_source_dirs; then
        log_error "Source directory check failed, build aborted"
        exit 1
    fi
    
    # 创建临时目录
    create_temp_dir
    
    # 设置退出时清理临时目录
    trap cleanup EXIT
    
    # 复制交付件
    copy_artifacts
    
    # 创建安装脚本
    create_install_script
    
    # 创建run文件
    if ! create_run_file; then
        log_error "Failed to create run file"
        exit 1
    fi
    
    # 显示构建完成信息
    show_build_info
}

# 脚本入口点
main "$@"