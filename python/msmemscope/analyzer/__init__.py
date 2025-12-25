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

from typing import Dict, Any, List
from .leaks import LeaksAnalyzer, LeaksConfig, check_leaks
from .inefficient import InefficientAnalyzer, InefficientConfig, check_inefficient
import inspect

# 分析器注册表
_ANALYZER_REGISTRY = {
    "leaks": LeaksAnalyzer,
    "inefficient": InefficientAnalyzer
}

# 配置类映射
_CONFIG_MAPPING = {
    "leaks": LeaksConfig,
    "inefficient": InefficientConfig
}


# 返回分析的结果 分析结果返回空 代表分析结束
def analyze(analyzer_type: str, **kwargs):
    if analyzer_type not in _ANALYZER_REGISTRY:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}. Available types are {list(_ANALYZER_REGISTRY.keys())}")
    
    # 创建对应的config对象
    config_class = _CONFIG_MAPPING[analyzer_type]
    config = config_class(**kwargs)

    # 执行分析
    analyzer = _ANALYZER_REGISTRY[analyzer_type]()

    return analyzer.analyze(config)


# 支持查询分析器类型和需要设置的配置器参数
def list_analyzers() -> List[str]:
    return list(_ANALYZER_REGISTRY.keys())


def get_analyzer_config(analyzer_type: str) -> Dict[str, Any]:
    if analyzer_type not in _CONFIG_MAPPING:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}. Available types are {list(_CONFIG_MAPPING.keys())}")

    config_class = _CONFIG_MAPPING[analyzer_type]
    signature = inspect.signature(config_class.__init__)

    return {
        "config_class": config_class.__name__,
        "parameters": list(signature.parameters.keys())[1:]
    }

# 导出对外接口
__all__ = [
    "analyze",
    "list_analyzers",
    "get_analyzer_config",
    "check_leaks",
    "check_inefficient",
    "LeaksAnalyzer",
    "InefficientAnalyzer",
    "LeaksConfig",
    "InefficientConfig"
]