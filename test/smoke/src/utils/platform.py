#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from enum import Enum, auto
from typing import Optional


class DeviceName(Enum):
    Ascend910B1 = auto()
    Ascend910B2 = auto()
    Ascend910B3 = auto()
    Ascend910B4 = auto()


def parse_device_name(name: str) -> Optional[DeviceName]:
    _device_name_map = {n.name: n for n in DeviceName}
    return _device_name_map.get(name, None)


class AiCoreArch(Enum):
    DAV_C220 = auto()
    DAV_C220_VEC = auto()
    DAV_C220_CUBE = auto()


def get_aicore_arch_str(aicore_arch: AiCoreArch) -> Optional[str]:
    _aicore_arch_str_map = {
        AiCoreArch.DAV_C220: "dav-c220",
        AiCoreArch.DAV_C220_VEC: "dav-c220-vec",
        AiCoreArch.DAV_C220_CUBE: "dav-c220-cube",
    }
    return _aicore_arch_str_map.get(aicore_arch, None)
