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

import logging
from ._msmemscope import _take_snapshot

import torch
import torch_npu

def get_device_memory_info(device=None):
    """
    Get memory information for the specified device.
    
    Args:
        device: Device index or None for current device
    
    Returns:
        dict: Memory information
    """
    if device is None:
        device = torch.device('npu')
    elif isinstance(device, int):
        device = torch.device('npu', device)
    
    try:
        memory_info = {
            "device": device.index if device.index is not None else 0,
            "memory_reserved": torch_npu.npu.memory_reserved(device),
            "max_memory_reserved": torch_npu.npu.max_memory_reserved(device),
            "memory_allocated": torch_npu.npu.memory_allocated(device),
            "max_memory_allocated": torch_npu.npu.max_memory_allocated(device),
        }
        
        # Get total and free memory
        free_mem, total_mem = torch_npu.npu.mem_get_info(device)
        memory_info["total_memory"] = total_mem
        memory_info["free_memory"] = free_mem
        
        return memory_info
    except ImportError:
        logging.warning("[msmemscope]: Import torch_npu failed when get device memory info! Please check it.")
        return {}

def take_snapshot(device=None, name="Memory Snapshot"):
    """
    Take a memory snapshot for the specified devices.
    
    Args:
        device: Device index, list/tuple of device indices, or None for all devices
        output: Output path for snapshot (default: C++ module's default path)
        name: Custom name for the snapshot event
    """
    
    # Determine devices to snapshot
    if device is None:
        # Get all available devices
        device_count = torch_npu.npu.device_count()
        devices = list(range(device_count))
    elif isinstance(device, int):
        # Single device
        devices = [device]
    elif isinstance(device, (list, tuple)):
        # Multiple devices
        devices = list(device)
    else:
        logging.warning("[msmemscope]: Invalid device parameter, using current device!")
        devices = [torch.device('npu').index if torch.device('npu').index is not None else 0]
    
    # Take snapshot for each device
    for dev in devices:
        memory_info = get_device_memory_info(dev)
        if memory_info:
            # Add output and name to memory info
            memory_info["name"] = name
            _take_snapshot(memory_info)
        else:
            logging.warning(f"[msmemscope]: Failed to get device {dev} memory info, snapshot skipped!")





