// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef OP_EXCUTE_WATCH_H
#define OP_EXCUTE_WATCH_H

#include <mutex>
#include <string>
#include <vector>
#include "atb_hooks/atb_stub.h"
#include "atb_hooks/mki_stub.h"
#include "event_report.h"
#include "record_info.h"
#include "tensor_dumper.h"
#include "tensor_monitor.h"

namespace Leaks {

class OpExcuteWatch {
public:
    OpExcuteWatch(const OpExcuteWatch&) = delete;
    OpExcuteWatch& operator=(const OpExcuteWatch&) = delete;

    static OpExcuteWatch& GetInstance()
    {
        static OpExcuteWatch instance;
        return instance;
    }

    void OpExcuteBegin(aclrtStream stream, const std::string &rawOp, OpType type);
    void OpExcuteEnd(aclrtStream stream, const std::string &rawOp, const std::vector<MonitoredTensor>& tensors,
        OpType type);
    void KernelExcute(aclrtStream stream, const std::string &rawKernel, const Mki::SVector<Mki::Tensor>& tensors,
        OpType type);

    std::string GetWatchedOpName();

private:
    OpExcuteWatch()
    {
        Config config = EventReport::Instance(CommType::SOCKET).GetConfig();
        fistWatchOp_ = std::string(config.watchConfig.start);
        lastWatchOp_ = std::string(config.watchConfig.end);
        outputId_ = config.watchConfig.outputId;
    };
    ~OpExcuteWatch() = default;

    // 落盘时需要用完整的opName，包含卡号和线程号。
    void BeginExcute(aclrtStream stream, const std::string &rawItem, OpType type);
    void EndExcute(aclrtStream stream, const std::string &excuteItem, const std::string &rawItem, OpType type,
        const std::vector<MonitoredTensor> &outputTensors = {}, uint32_t outputId = 0);

    bool IsFirstWatchOp(const std::string &op);
    bool IsLastWatchOp(const std::string &op);

    void SetWatchedOpName(const std::string &name);
    void ClearWatchedOpName();

    bool IsInMonitoring();

private:
    std::string watchedOpName_ {};
    std::string fistWatchOp_;
    std::string lastWatchOp_;
    uint32_t outputId_;
    std::mutex mutex_;
};

}
#endif