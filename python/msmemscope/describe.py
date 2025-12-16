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
 
from functools import wraps
from ._msmemscope import _describer


class Describe:
    def __call__(self, obj=None, owner=''):
        if obj is not None:
            if isinstance(obj, int):
                address = obj
            else:
                address = self._get_address(obj)
            _describer.describe_addr(address, owner)
            return None
        else:
            return DescribeContext(owner)

    @staticmethod
    def _get_address(tensor):
        try:
            return tensor.data_ptr()
        except AttributeError as error:
            raise NotImplementedError("Unsupported tensor type") from error


class DescribeContext:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        _describer.describe(self.owner)
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        _describer.undescribe(self.owner)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

describer = Describe()