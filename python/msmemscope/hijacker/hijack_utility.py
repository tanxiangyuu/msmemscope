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

import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from importlib.abc import Loader, MetaPathFinder
from importlib.util import module_from_spec, spec_from_loader
from uuid import uuid4

"""
支持完全替换、前置钩子和后置钩子三种劫持类型
"""
REPLACE = 0
PRE_HOOK = 1
POST_HOOK = 2
_ALL_ACTION = [REPLACE, PRE_HOOK, POST_HOOK]


class HijackHandler:
    def __init__(self, unit):
        self.unit = unit
        self.call_count = 0
        self.call_data = defaultdict(dict)
        self.released = False

"""
外部调用的劫持注册和取消函数:
(1)hijakcer:劫持注册声明
(2)release:取消劫持注册
"""
def hijacker(
    *, stub: callable, module: str, cls: str = "", function: str = "", action: int = REPLACE, priority: int = 100
) -> str:
    """
    Hijack module-import process or function execution process.
    Support attaching pre/post hooks to the process, or replacing function implementations.

    .. target::
        When only set "module": module
        When set "module" and "function": function in module
        When set "module", "cls" and "function": function in class

    .. warning::
        The pre-hook of the module-import process will only take effect if it is set before the module is imported.
        If the module is modified in its post-hook, the impact cannot be restored even if the hijacking is released.

    Parameters
    ----------
    stub: Callable object.
        Follow different format under different target and action.
        ---------------------------------------------------------------------------------------------------------------
        |  target  |  action   |           format             |                   description                         |
        |-------------------------------------------------------------------------------------------------------------|
        | module   | pre-hook  | callable()                   | Called before module import.                          |
        |-------------------------------------------------------------------------------------------------------------|
        | module   | post-hook | callable(m)                  | Called after module import. "m" is the module.        |
        |-------------------------------------------------------------------------------------------------------------|
        | function | replace   | ret = callable(*args, **kws) | Replace original object.                              |
        |-------------------------------------------------------------------------------------------------------------|
        | function | pre-hook  | args, kws =                  | Called before function execution, and the return will |
        |          |           |    callable(*args)           | replace original input of the target function.        |
        |-------------------------------------------------------------------------------------------------------------|
        | function | post-hook | ret = callable(ret, *args,   | Called after function execution, and the return will  |
        |          |           |           **kws)             | replace original return of target function.           |
        ---------------------------------------------------------------------------------------------------------------
    module: str
        Full name of target module.
    cls: str, optional
        Full name of target class.
    function: str, optional
        Name of target function.
    action: enum, optional
        Choose between REPLACE, PRE_HOOK, and POST_HOOK.
    priority: int, optional
        The smaller the value is, the higher the priority is. When multiple hooks are set on the same target, they will
        be excuted by priority.

    Returns
    -------
    hander:
        Handler to a hijacking. E.g., handler.unit, handler.call_data, handler.released.
    """
    HiJackerManager.initialize()
    unit = HijackerUnit(stub, module, cls, function, action, priority)
    handler = HijackHandler(unit)
    unit.handler = handler
    HiJackerManager.add_unit(unit)
    return handler

def release(handler):
    """
    Cancel a hijacking. "handler" is returned by function "hijack".
    """
    if isinstance(handler, HijackHandler):
        handler.released = True
        HiJackerManager.remove_unit(handler.unit)
    else:
        raise TypeError("Handler must be an instance of HijackHandler.")

class HijackerUnit:
    """
    HijackerUnit:单个劫持单元
    """
    def __init__(self, stub, module, cls, function, action, priority):
        self.stub = stub
        self.module = module
        self.cls = cls
        self.function = function
        self.action = action
        self.priority = priority
        self.target = f"{module}-{cls}-{function}"
        self.handler = None
        self._check_para_valid()

    def _check_para_valid(self):
        if not callable(self.stub):
            raise TypeError('"stub" should be callable.')
        if not self.module:
            raise ValueError('"module" is required and cannot be empty.')
        if not isinstance(self.module, str):
            raise TypeError('"module" should be a string.')
        if self.cls and not isinstance(self.cls, str):
            raise TypeError('"cls" should be a string.')
        if self.cls and not self.function:
            raise ValueError('"function" is required when "cls" is specified.')
        if self.function and not isinstance(self.function, str):
            raise TypeError('"function" should be a string.')
        if self.action not in _ALL_ACTION:
            raise ValueError(f'"action" should be one of {_ALL_ACTION}, got {self.action}.')
        if not self.cls and not self.function and self.action == REPLACE:
            raise ValueError('Replacement of an entire module is not supported.')
        if not isinstance(self.priority, int):
            raise TypeError('"priority" should be an integer.')

class HiJackerWrapperObj(ABC):
    """
    HiJackerWrapperObj:
    抽象基类,用于HiJackerWrapperModule(模块劫持)和HiJackerWrapperFunction(函数劫持)的实现
    """
    def __init__(self, name):
        self.name = name
        self.ori_obj = None
        self.replacement = []
        self.pre_hooks = []
        self.post_hooks = []
        self.mod_name, self.class_name, self.func_name = name.split("-")

    @property
    def is_empty(self):
        return not self.replacement and not self.pre_hooks and not self.post_hooks

    @abstractmethod
    def activate(self):
        pass

    @abstractmethod
    def deactivate(self):
        pass

    def add_unit(self, unit):
        if unit.action == REPLACE:
            self.replacement.append(unit)
            self.replacement.sort(key=lambda x: x.priority)
        elif unit.action == PRE_HOOK:
            self.pre_hooks.append(unit)
            self.pre_hooks.sort(key=lambda x: x.priority)
        else:
            self.post_hooks.append(unit)
            self.post_hooks.sort(key=lambda x: x.priority)

    def remove_unit(self, unit):
        if unit.action == REPLACE:
            self.replacement.remove(unit)
        elif unit.action == PRE_HOOK:
            self.pre_hooks.remove(unit)
        else:
            self.post_hooks.remove(unit)

    def set_ori_obj(self, obj):
        self.ori_obj = obj

class HiJackerWrapperModule(HiJackerWrapperObj):
    def __init__(self, name):
        super().__init__(name)

    def exec_pre_hook(self):
        for unit in self.pre_hooks:
            unit.stub()

    def exec_post_hook(self, m):
        self.set_ori_obj(m)
        for unit in self.post_hooks:
            unit.stub(m)

    def add_unit(self, unit):
        super().add_unit(unit)
        if unit.action == POST_HOOK:
            m = sys.modules.get(self.mod_name)
            if m:
                unit.stub(m)

    def activate(self):
        HiJackerPathFinder.add_mod(self.mod_name)

    def deactivate(self):
        HiJackerPathFinder.remove_mod(self.mod_name)

class HiJackerWrapperFunction(HiJackerWrapperObj):
    def __init__(self, name):
        super().__init__(name)
        self.mod_hijacker = None

    def activate(self):
        def replace_closure(class_name, func_name, wrapper):
            def modify_module(m):
                parent_obj = m
                class_chain = class_name.split(".") if class_name else []
                for c in class_chain:
                    if not hasattr(parent_obj, c):
                        return
                    parent_obj = getattr(parent_obj, c)
                if parent_obj and hasattr(parent_obj, func_name):
                    ori_obj = getattr(parent_obj, func_name)
                    self.set_ori_obj(ori_obj)
                    setattr(parent_obj, func_name, wrapper)
                return

            return modify_module

        self.mod_hijacker = hijacker(
            stub=replace_closure(self.class_name, self.func_name, self._get_wrapper()),
            module=self.mod_name,
            action=POST_HOOK,
            priority=0,
        )
        return

    def deactivate(self):
        if self.mod_hijacker:
            release(self.mod_hijacker)
            self.mod_hijacker = None
        mod = sys.modules.get(self.mod_name)
        if mod and self.ori_obj:
            parent_obj = mod
            class_chain = self.class_name.split(".") if self.class_name else []
            for c in class_chain:
                if not hasattr(parent_obj, c):
                    self.ori_obj = None
                    return
                parent_obj = getattr(parent_obj, c)
            if parent_obj and hasattr(parent_obj, self.func_name):
                setattr(parent_obj, self.func_name, self.ori_obj)
        self.ori_obj = None
        return

    def _get_wrapper(self):
        def wrapper(*args, **kws):
            if not self.ori_obj:
                raise RuntimeError(
                    "Original function object not found. Ensure activate() was called successfully."
                )
            # 执行前置钩子函数
            for unit in self.pre_hooks:
                result = unit.stub(*args, **kws)
                if isinstance(result, tuple):
                    args, kws = result
                else:
                    raise TypeError("Pre-hook must return a tuple of (args, kws)")
            # 执行实际函数
            f = self.replacement[0].stub if self.replacement else self.ori_obj
            ret = f(*args, **kws)
            # 执行后置钩子
            for unit in self.post_hooks:
                ret = unit.stub(ret, *args, **kws)
            return ret

        return wrapper

class HiJackerPathFinder(MetaPathFinder):
    """
    HiJackerPathFinder:Python模块导入劫持器,可以指定要劫持的模块,然后使用自定义的加载器来加载这些模块
    vllm的mock暂不涉及模块劫持,一般是劫持函数方法
    """
    _modules_of_interest = set()

    @classmethod
    def add_mod(cls, name):
        cls._modules_of_interest.add(name)

    @classmethod
    def remove_mod(cls, name):
        cls._modules_of_interest.discard(name)

    def find_spec(self, fullname, path, target=None):
        if fullname not in self._modules_of_interest:
            return None
        for finder in sys.meta_path:
            if isinstance(finder, HiJackerPathFinder):
                continue
            spec = finder.find_spec(fullname, path, target)
            if not spec:
                continue
            return spec_from_loader(fullname, HiJackerLoader(spec))
        return None

    def find_module(self, fullname, path=None):
        if fullname not in self._modules_of_interest:
            return None
        for finder in sys.meta_path:
            if isinstance(finder, HiJackerPathFinder):
                continue
            loader = finder.find_module(fullname, path)
            if not loader:
                continue
            return HiJackerLoader(spec_from_loader(fullname, loader))
        return None

class HiJackerManager:
    """
    所有hijacker的统一管理器
    """
    _initialized = False
    _hijacker_units = {}
    _hijacker_wrappers = {}

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return
        sys.meta_path.insert(0, HiJackerPathFinder())
        cls._initialized = True

    @classmethod
    def add_unit(cls, unit):
        handler = uuid4().hex
        cls._hijacker_units[handler] = unit
        wrapper_obj = cls._hijacker_wrappers.get(unit.target)
        if not wrapper_obj:
            wrapper_obj = cls._build_wrapper_obj(unit.target)
            cls._hijacker_wrappers[unit.target] = wrapper_obj
            wrapper_obj.activate()
        wrapper_obj.add_unit(unit)
        return handler

    @classmethod
    def remove_unit(cls, handler):
        unit = cls._hijacker_units.get(handler)
        if not unit:
            return
        wrapper_obj = cls._hijacker_wrappers.get(unit.target)
        wrapper_obj.remove_unit(unit)
        if wrapper_obj.is_empty:
            wrapper_obj.deactivate()
            del cls._hijacker_wrappers[unit.target]
        del cls._hijacker_units[handler]

    @classmethod
    def get_module_wrapper(cls, name):
        return cls._hijacker_wrappers.get(f"{name}--")

    @classmethod
    def _build_wrapper_obj(cls, name):
        _, _, f = name.split("-")
        if f:
            return HiJackerWrapperFunction(name)
        else:
            return HiJackerWrapperModule(name)

class HiJackerLoader(Loader):
    def __init__(self, ori_spec):
        self.ori_spec = ori_spec

    def create_module(self, spec):
        module = module_from_spec(self.ori_spec)
        return module

    def load_module(self, fullname):
        module = self.ori_spec.loader.load_module(fullname)
        return module

    def exec_module(self, module):
        wrapper = HiJackerManager.get_module_wrapper(module.__name__)
        if wrapper:
            wrapper.exec_pre_hook()
        self.ori_spec.loader.exec_module(module)
        if wrapper:
            wrapper.exec_post_hook(module)
