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

#include "loaded_so.h"

#include <cstring>
#include <fstream>
#include <string>

namespace MemScope
{

const char* const HOST_HOOK_SO_NAME = "libmsmemscope_host_mem_hook.so";

namespace
{
// NPU采集hook链(与wrapper LD_PRELOAD清单一致)
const char* const NPU_HOOK_SO_NAMES[] = {"libleaks_ascend_hal_hook.so", "libascend_mstx_hook.so",
                                         "libascend_kernel_hook.so", "libatb_abi_0_hook.so", "libatb_abi_1_hook.so"};
}  // namespace

bool IsSoLoaded(const char* soName)
{
    std::ifstream mapsFile("/proc/self/maps");
    if (!mapsFile.is_open())
    {
        return false;
    }
    std::string line;
    while (std::getline(mapsFile, line))
    {
        if (line.find(soName) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

bool IsNpuHookLoaded()
{
    for (const char* soName : NPU_HOOK_SO_NAMES)
    {
        if (IsSoLoaded(soName))
        {
            return true;
        }
    }
    return false;
}

}  // namespace MemScope
