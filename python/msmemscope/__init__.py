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
from .take_snapshot import TakeSnapshot

# 自定义打点trace事件
from .record_function import RecordFunction


from .hijacker.hijack_manager import memscope_hijack_manager
def init_framework_hooks(framework: str, version: str, component:str, hook_type: str):
    """
    init_framework_hooks:注册对应framework的所有默认hook函数钩子

    framework:支持框架(当前支持vllm_ascend,后续将支持verl/fsdp)
    version:框架对应版本
    component:指定要hook的组件或模块 vllm_ascend对应的worker,verl对应的actor_rollout\ref\critic\reward等
    hook_type:对应的hook函数(decompose:显存拆解,sanpshot:显存快照)
    """
    return memscope_hijack_manager.init_framework_hooks(framework, version, component, hook_type)

def cleanup_framework_hooks():
    """
    cleanup_framework_hooks:清除对应framework的所有默认hook函数钩子
    """
    return memscope_hijack_manager.cleanup_framework_hooks() 
