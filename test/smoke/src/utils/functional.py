#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from functools import wraps


def uncurry(f):
    @wraps(f)
    def _wrapper(args):
        return f(*args)
    return _wrapper


def curry(f):
    @wraps(f)
    def _wrapper(*args):
        return f(args)
    return _wrapper


def as_arg(arg):
    def _wrapper(f):
        return f(arg)
    return _wrapper
