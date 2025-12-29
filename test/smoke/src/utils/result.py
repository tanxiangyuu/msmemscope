#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from dataclasses import dataclass
from typing import Any


@dataclass
class Result:
    success: bool
    expected: Any
    got: Any


def join(lhs: Result, rhs: Result) -> Result:
    if (lhs.success):
        return rhs
    if (rhs.success):
        return lhs
    return Result(False, (lhs.expected, rhs.expected), (lhs.got, rhs.got))
