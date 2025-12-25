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

import os
import ctypes

# 首先设置环境变量
ASCEND_HOME_PATH = os.getenv('ASCEND_HOME_PATH')
LEAKS_LIB_PATH = ""
if ASCEND_HOME_PATH:
    LEAKS_LIB_PATH = os.path.join(ASCEND_HOME_PATH, "tools", "msmemscope", "lib64")

# 加载依赖
if LEAKS_LIB_PATH:
    LEAKS_SO_PATH = os.path.join(LEAKS_LIB_PATH, "libascend_leaks.so")
    try:
        ctypes.CDLL(LEAKS_SO_PATH, mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        print(f"[ERROR] {e}")

from ._msmemscope import _watcher
from ._msmemscope import _tracer
from ._msmemscope import start, stop, step, config

tracer = _tracer
watcher = _watcher

from .analyzer import (
    analyze,
    list_analyzers,
    get_analyzer_config,
    check_leaks,
    check_inefficient
)