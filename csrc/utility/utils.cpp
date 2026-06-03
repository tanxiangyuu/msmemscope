/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#include <unistd.h>

#include <cstdint>
#include <cstdio>

namespace Utility
{

uint64_t GetProcessVmRss()
{
    FILE *fp = fopen("/proc/self/statm", "r");
    if (fp == nullptr)
    {
        return 0;
    }

    uint32_t vms_pages;
    uint32_t rss_pages;
    if (fscanf(fp, "%u %u", &vms_pages, &rss_pages) != 2)
    {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    static const uint32_t page_size = static_cast<uint32_t>(sysconf(_SC_PAGESIZE));
    return static_cast<uint64_t>(rss_pages) * page_size;
}
}  // namespace Utility
