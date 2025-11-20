# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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