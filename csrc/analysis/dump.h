// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef DUMP_H
#define DUMP_H

#include <unordered_map>
#include <mutex>

#include "analyzer_base.h"
#include "config_info.h"
#include "data_handler.h"

namespace Leaks {

const std::unordered_map<EventBaseType, std::string> EVENT_BASE_TYPE_MAP = {
    {EventBaseType::MALLOC, "MALLOC"},
    {EventBaseType::ACCESS, "ACCESS"},
    {EventBaseType::FREE, "FREE"},
    {EventBaseType::MSTX, "MSTX"},
    {EventBaseType::OP_LAUNCH, "OP_LAUNCH"},
    {EventBaseType::KERNEL_LAUNCH, "KERNEL_LAUNCH"},
    {EventBaseType::SYSTEM, "SYSTEM"},
    {EventBaseType::CLEAN_UP, "CLEAN_UP"},
};

const std::unordered_map<EventSubType, std::string> EVENT_SUB_TYPE_MAP = {
    {EventSubType::PTA_CACHING, "PTA"},  // 兼容性考虑，对外展示保持不变
    {EventSubType::PTA_WORKSPACE, "PTA_WORKSPACE"},
    {EventSubType::ATB, "ATB"},
    {EventSubType::MINDSPORE, "MINDSPORE"},
    {EventSubType::HAL, "HAL"},
    {EventSubType::HOST, "HAL"},
    {EventSubType::ATB_READ, "READ"},
    {EventSubType::ATB_WRITE, "WRITE"},
    {EventSubType::ATB_READ_OR_WRITE, "UNKNOWN"},
    {EventSubType::ATEN_READ, "READ"},
    {EventSubType::ATEN_WRITE, "WRITE"},
    {EventSubType::ATEN_READ_OR_WRITE, "UNKNOWN"},
    {EventSubType::ATB_START, "ATB_START"},
    {EventSubType::ATB_END, "ATB_END"},
    {EventSubType::ATEN_START, "ATEN_START"},
    {EventSubType::ATEN_END, "ATEN_END"},
    {EventSubType::KERNEL_LAUNCH, "KERNEL_LAUNCH"},
    {EventSubType::KERNEL_EXECUTE_START, "KERNEL_EXECUTE_START"},
    {EventSubType::KERNEL_EXECUTE_END, "KERNEL_EXECUTE_END"},
    {EventSubType::ATB_KERNEL_START, "KERNEL_START"},
    {EventSubType::ATB_KERNEL_END, "KERNEL_END"},
    {EventSubType::ACL_INIT, "ACL_INIT"},
    {EventSubType::ACL_FINI, "ACL_FINI"},
    {EventSubType::MSTX_MARK, "Mark"},
    {EventSubType::MSTX_RANGE_START, "Range_start"},
    {EventSubType::MSTX_RANGE_END, "Range_end"},
    {EventSubType::CLEAN_UP, "CLEAN_UP"},
};

class Dump : public AnalyzerBase {
public:
    static Dump& GetInstance(Config config);
    void EventHandle(std::shared_ptr<EventBase>& event, MemoryState* state) override;
private:
    explicit Dump(Config config);
    ~Dump() override = default;
    Dump(const Dump&) = delete;
    Dump& operator=(const Dump&) = delete;
    Dump(Dump&& other) = delete;
    Dump& operator=(Dump&& other) = delete;

    void DumpMemEventBeforeMalloc(MemoryState* state);
    void DumpMemEventAfterFree(MemoryState* state);
    void DumpMemEventBeforeCleanUp(std::shared_ptr<CleanUpEvent>& event);
    void DumpMemoryEvent(std::shared_ptr<MemoryEvent>& event, MemoryState* state);

    void DumpMstxEvent(std::shared_ptr<MstxEvent>& event);
    void DumpOpLaunchEvent(std::shared_ptr<OpLaunchEvent>& event);
    void DumpKernelLaunchEvent(std::shared_ptr<KernelLaunchEvent>& event);
    void DumpSystemEvent(std::shared_ptr<SystemEvent>& event);

    Config config_;
    std::mutex fileMutex_;
    std::unique_ptr<DataHandler> handler_;
};

}

#endif