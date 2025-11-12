// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef DUMP_H
#define DUMP_H

#include <unordered_map>
#include <mutex>

#include "analyzer_base.h"
#include "config_info.h"
#include "data_handler.h"

namespace Leaks {

class Dump : public AnalyzerBase {
public:
    static Dump& GetInstance(Config config);
    void EventHandle(std::shared_ptr<EventBase>& event, MemoryState* state) override;
private:
    explicit Dump(Config config);
    ~Dump() override;
    Dump(const Dump&) = delete;
    Dump& operator=(const Dump&) = delete;
    Dump(Dump&& other) = delete;
    Dump& operator=(Dump&& other) = delete;

    void DumpMemoryState(MemoryState* state);
    void DumpMemoryEvent(std::shared_ptr<MemoryEvent>& event, MemoryState* state);

    void DumpMstxEvent(std::shared_ptr<MstxEvent>& event);
    void DumpOpLaunchEvent(std::shared_ptr<OpLaunchEvent>& event);
    void DumpKernelLaunchEvent(std::shared_ptr<KernelLaunchEvent>& event);
    void DumpSystemEvent(std::shared_ptr<SystemEvent>& event);

    void WriteToFile(const std::shared_ptr<EventBase>& event);

    Config config_;
    std::unordered_map<std::string, std::unique_ptr<DataHandler>> handlerMap_;
    std::vector<std::shared_ptr<EventBase>> sharedEventLists_;
};

}

#endif