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
from typing import Dict, List

from .._msmemscope import _describer
from ..take_snapshot import take_snapshot

DEFAULT_PRIORITY = 100


class MemScopeHijackUnit:
    """
    最小的Hook单元,封装单个劫持目标的前置/后置钩子逻辑
    """
    def __init__(self, hook_type: str, module: str, class_name: str, method_name: str, priority: int = DEFAULT_PRIORITY):
        """
        初始化Hook单元
        :param hook_type: 钩子功能类型(如 decompose/snapshot)
        :param module: 目标模块名(如 vllm_ascend.worker.model_runner_v1)
        :param class_name: 目标类名(如 NPUModelRunner)
        :param method_name: 目标方法名(如 load_model)
        :param priority: 钩子优先级(数值越小优先级越高)
        """
        self.hook_type = hook_type
        self.module = module        
        self.class_name = class_name      
        self.method_name = method_name     
        self.priority = priority      
        self.identifier = f"{self.module}@{self.class_name}@{self.method_name}"

    def prehook_func(self, *args, **kwargs):
        if self.hook_type == "decompose":
            _describer.describe(self.identifier)
        elif self.hook_type == "snapshot":
            take_snapshot(None, f"{self.identifier}@Start")
        else:
            pass
        return args, kwargs

    def posthook_func(self, ret, *args, **kwargs):
        if self.hook_type == "decompose":
            _describer.undescribe(self.identifier)
        elif self.hook_type == "snapshot":
            take_snapshot(None, f"{self.identifier}@End")
        else:
            pass
        return ret


class MemScopeHijackMap:
    """
    单例类:维护framework/version/component/hook_type四级劫持映射,提供劫持单元列表生成能力
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化核心映射表:framework → version → component → hook_type → 劫持目标列表"""
        self.hijack_mapping: Dict[str, Dict[str, Dict[str, Dict[str, List[List[str]]]]]] = {
            "vllm_ascend": {
                "11.0": {
                    "worker": {
                        "decompose": [
                            ["vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "load_model"],
                            ["vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "profile_run"],
                            ["vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "initialize_kv_cache"],
                            ["vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "execute_model"],
                        ],
                        "snapshot": [
                            ["vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "load_model"],
                            ["vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "profile_run"],
                            ["vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "initialize_kv_cache"],
                            ["vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "execute_model"],
                        ]
                    }
                }
            },
            "verl": {
                "0.7.0":{
                    "TaskRunner":{
                        "snapshot": [
                            ["verl.trainer.main_ppo","TaskRunner","add_actor_rollout_worker"],
                            ["verl.trainer.main_ppo","TaskRunner","add_critic_worker"],
                            ["verl.trainer.main_ppo","TaskRunner","add_reward_model_worker"],
                            ["verl.trainer.main_ppo","TaskRunner","add_ref_policy_worker"],
                            ["verl.trainer.main_ppo","TaskRunner","run"],
                        ]
                    }
                }
            }
            
        }

    def get_hijack_entries(self, framework: str, version: str, component: str, hook_type: str) -> List[List[str]]:
        """
        获取指定框架/版本/组件/钩子类型对应的劫持目标列表
        :param framework: 框架名(如 vllm_ascend/pytorch)
        :param version: 框架版本(如 11.0)
        :param component: 组件名(如 model_runner/memory_manager)
        :param hook_type: 钩子类型(如 decompose/snapshot)
        :return: 劫持目标列表 [[module, class_name, method_name], ...]
        """
        if framework not in self.hijack_mapping:
            print(f"[msmemscope] Error: 框架 '{framework}' 不存在！当前支持的框架: {list(self.hijack_mapping.keys())}")
            return []

        version_map = self.hijack_mapping[framework]
        if version not in version_map:
            print(f"[msmemscope] Error: 框架 '{framework}' 下的版本 '{version}' 不存在！当前支持的版本: {list(version_map.keys())}")
            return []

        component_map = version_map[version]
        if component not in component_map:
            print(f"[msmemscope] Error: 框架 '{framework}' 版本 '{version}' 下的组件 '{component}' 不存在！当前支持的组件: {list(component_map.keys())}")
            return []

        hook_type_map = component_map[component]
        if hook_type not in hook_type_map:
            print(f"[msmemscope] Error: 框架 '{framework}' 版本 '{version}' 组件 '{component}' 下的钩子类型 '{hook_type}' 不存在！当前支持的类型: {list(hook_type_map.keys())}")
            return []

        return hook_type_map[hook_type]

    def get_hijack_unit_list(self, framework: str, version: str, component: str, hook_type: str) -> List[MemScopeHijackUnit]:
        """
        生成指定框架/版本/组件/钩子类型对应的MemScopeHijackUnit列表
        """
        hijack_unit_list = []
        hijack_entries = self.get_hijack_entries(framework, version, component, hook_type)

        for module, class_name, method_name in hijack_entries:
            hijack_unit = MemScopeHijackUnit(
                hook_type=hook_type,
                module=module,
                class_name=class_name,
                method_name=method_name
            )
            hijack_unit_list.append(hijack_unit)

        return hijack_unit_list


# 全局单例实例
memscope_hijack_map = MemScopeHijackMap()