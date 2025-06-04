// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "op_excute_watch.h"
#include "log.h"

namespace Leaks {

void OpExcuteWatch::BeginExcute(const std::string &rawItem, OpType type)
{
    OpEventType opEventType;
    if (type == OpType::ATB) {
        opEventType = OpEventType::ATB_START;
    } else if (type == OpType::ATEN) {
        opEventType = OpEventType::ATEN_START;
    } else {
        LOG_WARN("Get unknown type!");
        return ;
    }
    if (IsInMonitoring()) {
        TensorDumper::GetInstance().Dump(rawItem, opEventType);
        return;
    }
    return;
}

void OpExcuteWatch::EndExcute(const std::string &excuteItem, const std::string &rawItem, OpType type,
    const std::vector<MonitoredTensor> &outputTensors)
{
    OpEventType opEventType;
    if (type == OpType::ATB) {
        opEventType = OpEventType::ATB_END;
    } else if (type == OpType::ATEN) {
        opEventType = OpEventType::ATEN_END;
    } else {
        LOG_WARN("Get unknown type!");
        return ;
    }
    if (IsFirstWatchOp(excuteItem)) {
        SetWatchedOpName(excuteItem);
        TensorMonitor::GetInstance().AddWatchTensor(outputTensors);
        TensorDumper::GetInstance().Dump(rawItem, opEventType);

        return;
    }
    if (IsLastWatchOp(excuteItem)) {
        TensorDumper::GetInstance().Dump(rawItem, opEventType);
        
        ClearWatchedOpName();
        TensorMonitor::GetInstance().ClearCmdWatchTensor();

        return;
    }
    if (IsInMonitoring()) {
        TensorDumper::GetInstance().Dump(rawItem, opEventType);
        return;
    }

    return;
}

void OpExcuteWatch::OpExcuteBegin(const std::string &rawOp, OpType type)
{
    std::lock_guard<std::mutex> guard(mutex_);
    return BeginExcute(rawOp, type);
}

void OpExcuteWatch::OpExcuteEnd(const std::string &rawOp, const std::vector<MonitoredTensor>& tensors, OpType type)
{
    std::lock_guard<std::mutex> guard(mutex_);
    auto op = rawOp.substr(rawOp.find("/") + 1);
    if (!IsFirstWatchOp(op)) {
        return EndExcute(op, rawOp, type);
    }
    if (outputId_ < tensors.size()) {
        std::vector<MonitoredTensor> dumpTensors;
        MonitoredTensor tensor = tensors[outputId_];
        dumpTensors.emplace_back(tensor);
        return EndExcute(op, rawOp, type, dumpTensors);
    }
    return EndExcute(op, rawOp, type, tensors);
}

void OpExcuteWatch::KernelExcute(const std::string &rawKernel, const Mki::SVector<Mki::Tensor>& tensors, OpType type)
{
    std::lock_guard<std::mutex> guard(mutex_);
    auto beforPos = rawKernel.find("/before");
    auto afterPos = rawKernel.find("/after");
    if (beforPos != std::string::npos) {
        return BeginExcute(rawKernel.substr(0, beforPos), type);
    } else if (afterPos != std::string::npos) {
        std::string kernelDir = rawKernel.substr(0, afterPos).substr(rawKernel.find("/") + 1);
        if (!IsFirstWatchOp(kernelDir)) {
            return EndExcute(kernelDir, rawKernel.substr(0, afterPos), type);
        }
        std::vector<MonitoredTensor> dumpTensors;
        if (outputId_ < tensors.size()) {
            MonitoredTensor tensor {};
            tensor.data = tensors[outputId_].data;
            tensor.dataSize = static_cast<uint64_t>(tensors[outputId_].dataSize);
            dumpTensors.emplace_back(tensor);
            return EndExcute(kernelDir, rawKernel.substr(0, afterPos), type, dumpTensors);
        }
        for (auto &item : tensors) {
            MonitoredTensor tensor {};
            tensor.data = item.data;
            tensor.dataSize = static_cast<uint64_t>(item.dataSize);
            dumpTensors.emplace_back(tensor);
        }
        return EndExcute(kernelDir, rawKernel.substr(0, afterPos), type, dumpTensors);
    } else {
        CLIENT_ERROR_LOG("Invalid kernel info.\n");
        return;
    }
}

bool OpExcuteWatch::IsFirstWatchOp(const std::string &op)
{
    return op == fistWatchOp_;
}

bool OpExcuteWatch::IsLastWatchOp(const std::string &op)
{
    return op == lastWatchOp_;
}

void OpExcuteWatch::SetWatchedOpName(const std::string &name)
{
    watchedOpName_ = name;
}

std::string OpExcuteWatch::GetWatchedOpName()
{
    return watchedOpName_;
}

void OpExcuteWatch::ClearWatchedOpName()
{
    watchedOpName_ = "";
}

bool OpExcuteWatch::IsInMonitoring()
{
    if (TensorMonitor::GetInstance().GetCmdWatchedTensorsMap().size() == 0 &&
        TensorMonitor::GetInstance().GetPythonWatchedTensorsMap().size() == 0) {
        return false;
    } else {
        return true;
    }
}

}