// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef ATB_OP_EXCUTE_WATCH_H
#define ATB_OP_EXCUTE_WATCH_H

#include <mutex>
#include <string>
#include <vector>
#include "atb_tensor_dump.h"
#include "atb_hooks/atb_stub.h"
#include "atb_hooks/mki_stub.h"
#include "event_report.h"
#include "record_info.h"

namespace Leaks {

class ATBOpExcuteWatch {
public:
    ATBOpExcuteWatch(const ATBOpExcuteWatch&) = delete;
    ATBOpExcuteWatch& operator=(const ATBOpExcuteWatch&) = delete;

    static ATBOpExcuteWatch& GetInstance()
    {
        static ATBOpExcuteWatch instance;
        return instance;
    }

    void AtbOpExcuteBegin(const std::string &rawOp);
    void AtbOpExcuteEnd(const std::string &rawOp, const atb::SVector<atb::Tensor>& tensors);
    void AtbKernelExcute(const std::string &rawKernel, const Mki::SVector<Mki::Tensor>& tensors);

private:
    ATBOpExcuteWatch()
    {
        Config config = EventReport::Instance(CommType::SOCKET).GetConfig();
        fistWatchOp_ = std::string(config.watchConfig.start);
        lastWatchOp_ = std::string(config.watchConfig.end);
        outputId_ = config.watchConfig.outputId;
    };
    ~ATBOpExcuteWatch() = default;
    
    void DumpATBTensor(const std::string &op, OpEventType eventType);

    // 落盘时需要用完整的opName，包含卡号和线程号。
    void BeginExcute(const std::string &rawItem);
    void EndExcute(const std::string &excuteItem, const std::string &rawItem,
        const std::vector<Tensor> &outputTensors = {});

    bool IsFirstWatchOp(const std::string &op);
    bool IsLastWatchOp(const std::string &op);

    void SetWatchedTensors(const std::vector<Tensor> &tensors);
    std::vector<Tensor>& GetWatchedTensors();
    void ClearWatchedTensors();

    void SetWatchedOpName(const std::string &name);
    std::string GetWatchedOpName();
    void ClearWatchedOpName();

    bool IsInMonitoring();
    void SetMonitoringStatus();
    void UnSetMonitoringStatus();

private:
    bool inMonitoring_ = false;
    std::vector<Tensor> watchedTensors_ {};
    std::string watchedOpName_ {};
    std::string fistWatchOp_;
    std::string lastWatchOp_;
    uint32_t outputId_;

    std::mutex mutex_;
};

}
#endif