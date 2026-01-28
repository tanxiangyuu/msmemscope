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

from ._msmemscope import _watcher
from ._msmemscope import _tracer
from ._msmemscope import start, stop, step, config
from .utils import import_with_optional_deps

tracer = _tracer
watcher = _watcher

from .analyzer import (
    analyze,
    list_analyzers,
    get_analyzer_config,
    check_leaks,
    check_inefficient
)

# 指定需要的依赖包
take_snapshot = import_with_optional_deps(
    "take_snapshot", 
    "take_snapshot", 
    ["torch", "torch_npu"]
)