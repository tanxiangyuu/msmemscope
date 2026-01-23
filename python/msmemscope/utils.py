# -------------------------------------------------------------------------#
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.#
# Licensed under the Mulan PSL v2.#
# You can use this software according to the terms and conditions of the Mulan PSL v2.#
# You may obtain a copy of Mulan PSL v2 at:#
#     http://license.coscl.org.cn/MulanPSL2#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.#
# See the Mulan PSL v2 for more details.#
# -------------------------------------------------------------------------#

import importlib
import logging


def check_packages(packages, error_msg="Please check it."):
    """
    Check if required packages are installed.
    
    Args:
        packages (list): List of package names to check
        error_msg (str): Error message to display when import fails
    """
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            logging.warning(f"[msmemscope]: Import {package} failed! {error_msg}")

def import_with_optional_deps(module_name, func_name, required_packages, error_msg=None):
    """
    尝试导入需要可选依赖的模块或函数
    
    Args:
        module_name: 模块名称
        func_name: 函数名称
        required_packages: 需要的依赖包列表
        error_msg: 错误信息
        
    Returns:
        导入的函数或一个占位函数（当导入失败时）
    """
    # 检查所有需要的依赖包
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        # 如果有缺失的依赖包，定义占位函数
        if not error_msg:
            error_msg = f"[msmemscope]: Please install required packages: {', '.join(missing_packages)} for {func_name} functionality!"
        
        def placeholder(*args, **kwargs):
            raise ImportError(error_msg)
        return placeholder
    else:
        # 如果所有依赖包都已安装，导入并返回函数
        try:
            # 使用绝对导入方式，从msmemscope包开始
            full_module_name = f"msmemscope.{module_name}"
            module = importlib.import_module(full_module_name)
            return getattr(module, func_name)
        except ImportError as e:
            def placeholder(*args, **kwargs):
                raise ImportError(f"[msmemscope]: Failed to import {func_name}: {str(e)}")
            return placeholder
