# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from abc import ABC, abstractmethod
import stat
from pathlib import Path
import os

MAX_FILE_SIZE = 8 * 1024 * 1024 * 1024              # 支持最大文件大小为8G
INPUT_STR_MAX_LEN = 4096                            # 支持的最大文件路径长度


class AnalysisConfig:
    def __init__(self, input_path: str):
        self.input_path = input_path 
        
    def __post_init__(self):
        """包含对于输入文件路径的公共检查：包含文件存在性,可读性,文件大小,路径长度,是否为软链接,权限校验(group和other用户组不可写,属主为root或当前用户"""
        path = Path(self.input_path)
        current_uid = os.getuid()
        is_root = current_uid == 0
        file_stat = path.stat()

        # 对root用户添加风险提示
        if is_root:
            print("WARN: Running as root user: Some permission checks may be bypassed.")
        if not is_root:
            if file_stat.st_uid not in (0, current_uid):
                raise PermissionError(f"Must be owned by root (UID 0) or current user (UID {current_uid}) ")
        else:
            if file_stat.st_uid != 0:
                print(f"WARN: File {path} is not owned by root (current owner UID {file_stat.st_uid})")
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path} or not a regular file")
        if not os.access(str(path), os.R_OK):
            raise PermissionError(f"Read permission denied: {path}")
        if file_stat.st_size > MAX_FILE_SIZE:
            raise ValueError(f"Max allowed size is {MAX_FILE_SIZE / (1024*1024*1024):.2f}GB")
        if len(str(path)) > INPUT_STR_MAX_LEN:
            raise ValueError(f"Max allowed length is {INPUT_STR_MAX_LEN} characters")
        if path.is_symlink():
            raise ValueError(f"Unsupported file type: {path} is a symbolic link")
        if (file_stat.st_mode & stat.S_IWGRP) or (file_stat.st_mode & stat.S_IWOTH):
            raise PermissionError(f"Group or others have write permission: {path}. ")
        if not path.is_absolute():
            raise ValueError(f"File path must be absolute: {path}")
        # 目前只涉及csv和db文件的操作
        valid_extensions = {'.csv', '.db'}
        file_ext = path.suffix.lower()
        if file_ext not in valid_extensions:
            raise ValueError(f"Unsupported file type: {file_ext}.Only files ending with .csv or .db are supported.")


class BaseAnalyzer(ABC):
    """分析器基类:必须实现analyze"""

    @abstractmethod
    def analyze(self, config: AnalysisConfig):
        pass
