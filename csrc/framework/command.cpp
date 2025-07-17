// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "command.h"
#include <map>
#include "process.h"
#include "utils.h"
#include "analysis/memory_compare.h"
#include "analysis/dump_record.h"
#include "analysis/device_manager.h"

#include <iostream>

namespace Leaks {

void DumpDataCleanUp()
{
    Config config;
    auto memStateRecordMap = DeviceManager::GetInstance(config).GetMemoryStateRecordMap();
    for (auto it = memStateRecordMap.begin(); it != memStateRecordMap.end();) {
        auto memStateRecord = it->second;
        auto ptrMemoryInfoMap = memStateRecord->GetPtrMemInfoMap();
        for (auto itMemInfo = ptrMemoryInfoMap.begin(); itMemInfo != ptrMemoryInfoMap.end();) {
            auto memInfoLists = itMemInfo->second;
            auto key = itMemInfo->first;
            for (auto& memInfo : memInfoLists) {
                if (memInfo.container.event == "MALLOC") {
                    DumpRecord::GetInstance(config).SetAllocAttr(memInfo);
                }
                DumpRecord::GetInstance(config).WriteToFile(memInfo.container, memInfo.stack);
            }
            ++itMemInfo;
        }
        ++it;
    }
}

void Command::Exec() const
{
    LOG_INFO("Msleaks starts executing commands");

    if (userCommand_.config.enableCompare) {
        MemoryCompare::GetInstance(userCommand_.config).RunComparison(userCommand_.inputPaths);
        return;
    }

    // atexit注册资源清理回调函数, 清理函数涉及的类必须在该函数前构造
    DumpRecord::GetInstance(userCommand_.config);
    DeviceManager::GetInstance(userCommand_.config);
    std::atexit(DumpDataCleanUp);
    
    Process process(userCommand_.config);
    process.Launch(userCommand_.cmd);

    return;
}

}