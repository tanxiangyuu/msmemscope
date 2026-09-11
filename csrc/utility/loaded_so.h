/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#ifndef LOADED_SO_H
#define LOADED_SO_H

namespace MemScope
{
// host钩子so名(display hook输出与config() preload校验共用)
extern const char* const HOST_HOOK_SO_NAME;

// /proc/self/maps扫描判断目标so是否已加载(display hook与config() preload校验共用)
bool IsSoLoaded(const char* soName);
// 任一NPU钩子so已加载(hal/mstx/kernel/atb)
bool IsNpuHookLoaded();

}  // namespace MemScope

#endif
