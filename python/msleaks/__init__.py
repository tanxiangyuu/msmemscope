# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from ._msleaks import _watcher
from ._msleaks import _tracer
from ._msleaks import start, stop, config

from .leak_checker import _leakchecker
from .inefficient import inefficient_inner

tracer = _tracer
watcher = _watcher
leakchecker = _leakchecker
inefficient = inefficient_inner
