# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import sys
import logging
from typing import List, Tuple, Dict, Any
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import register_optimizer_step_post_hook
from ._msmemscope import _report_tensor


def append_tensor_info(
    tensor_info_list: List[Tuple[int, str]],
    tensor: torch.Tensor,
    category: str,
) -> None:
    if 'npu' in str(tensor.device).lower():
        tensor_info_list.append(tuple((tensor.data_ptr(), category)))
    return


def process_param(tensor_info_list: List[Tuple[int, str]], param: torch.Tensor, opt: Optimizer):
    append_tensor_info(tensor_info_list, param, "@model@weight")
    if param.grad is not None:
        append_tensor_info(tensor_info_list, param.grad, "@model@gradient")
    
    if param in opt.state:
        for _, state in opt.state[param].items():
            if torch.is_tensor(state):
                append_tensor_info(tensor_info_list, state, "@model@optimizer_state")


def global_optimizer_step_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
    tensor_info_list: List[Tuple[int, str]] = []

    for param_group in opt.param_groups:
        for param in param_group['params']:
            process_param(tensor_info_list, param, opt)
    
    _report_tensor.report_tensor(tensor_info_list)


class OptimizerStepHook:
    def __init__(self):
        self.global_handle = None
        self.enabled = False
    
    def __del__(self):
        if (sys is not None) and (not sys.is_finalizing()) and self.enable:
            self.disable()
    
    def enable(self):
        self.global_handle = register_optimizer_step_post_hook(global_optimizer_step_hook)
        self.enabled = True
    
    def disable(self):
        if self.global_handle is not None:
            self.global_handle.remove()
        self.enabled = False


def enable_optimizer_step_hook():
    optimizer_step_hook.enable()


def disable_optimizer_step_hook():
    optimizer_step_hook.disable()


logging.info(f"[msmemscope] Enable optimizer step hook.")
optimizer_step_hook = OptimizerStepHook()