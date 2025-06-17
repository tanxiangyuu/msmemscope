// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "op_excute_watch.h"
#include "client_process.h"

namespace Leaks {

uint64_t OpExcuteWatch::CountOpName(const std::string& name)
{
    std::lock_guard<std::mutex> guard(opNameMutex_);
    if (opNameCnt_.find(name) == opNameCnt_.end()) {
        opNameCnt_[name] = 0;
    } else {
        opNameCnt_[name] += 1;
    }
    return opNameCnt_[name];
}

void OpExcuteWatch::BeginExcute(aclrtStream stream, const std::string &rawItem, OpType type)
{
    OpEventType opEventType;
    if (type == OpType::ATB) {
        opEventType = OpEventType::ATB_START;
    } else if (type == OpType::ATEN) {
        opEventType = OpEventType::ATEN_START;
    } else {
        CLIENT_WARN_LOG("Get unknown type!");
        return ;
    }
    if (IsInMonitoring()) {
        std::string opName = rawItem + "_" + std::to_string(CountOpName(rawItem));
        TensorDumper::GetInstance().Dump(stream, opName, opEventType);
        return;
    }
    return;
}

void OpExcuteWatch::EndExcute(aclrtStream stream, const std::string &excuteItem, const std::string &rawItem,
    OpType type, const std::vector<MonitoredTensor> &outputTensors,  uint32_t outputId)
{
    OpEventType opEventType;
    if (type == OpType::ATB) {
        opEventType = OpEventType::ATB_END;
    } else if (type == OpType::ATEN) {
        opEventType = OpEventType::ATEN_END;
    } else {
        CLIENT_WARN_LOG("Get unknown type!");
        return ;
    }
    std::string opName = rawItem;
    if (IsFirstWatchOp(excuteItem) && watchedOpName_.empty()) {
        opName += "_" + std::to_string(CountOpName(rawItem));
        SetWatchedOpName(excuteItem);
        TensorMonitor::GetInstance().AddWatchTensor(outputTensors, outputId);
        TensorDumper::GetInstance().Dump(stream, opName, opEventType);

        return;
    }
    if (IsLastWatchOp(excuteItem)) {
        opName += "_" + std::to_string(opNameCnt_[rawItem]);
        TensorDumper::GetInstance().Dump(stream, opName, opEventType);
        
        ClearWatchedOpName();
        TensorMonitor::GetInstance().ClearCmdWatchTensor();

        return;
    }
    if (IsInMonitoring()) {
        opName += "_" + std::to_string(opNameCnt_[rawItem]);
        TensorDumper::GetInstance().Dump(stream, opName, opEventType);
        return;
    }

    return;
}

void OpExcuteWatch::OpExcuteBegin(aclrtStream stream, const std::string &rawOp, OpType type)
{
    std::lock_guard<std::mutex> guard(mutex_);
    return BeginExcute(stream, rawOp, type);
}

void OpExcuteWatch::OpExcuteEnd(aclrtStream stream,
    const std::string &rawOp, const std::vector<MonitoredTensor>& tensors, OpType type)
{
    std::lock_guard<std::mutex> guard(mutex_);
    auto op = rawOp.substr(rawOp.find("/") + 1);
    if (!IsFirstWatchOp(op)) {
        return EndExcute(stream, op, rawOp, type);
    }
    if (outputId_ < tensors.size()) {
        std::vector<MonitoredTensor> dumpTensors;
        MonitoredTensor tensor = tensors[outputId_];
        dumpTensors.emplace_back(tensor);
        return EndExcute(stream, op, rawOp, type, dumpTensors, outputId_);
    }
    return EndExcute(stream, op, rawOp, type, tensors);
}

void OpExcuteWatch::KernelExcute(aclrtStream stream,
    const std::string &rawKernel, const Mki::SVector<Mki::Tensor>& tensors, OpType type)
{
    std::lock_guard<std::mutex> guard(mutex_);
    auto beforPos = rawKernel.find("/before");
    auto afterPos = rawKernel.find("/after");
    if (beforPos != std::string::npos) {
        return BeginExcute(stream, rawKernel.substr(0, beforPos), type);
    } else if (afterPos != std::string::npos) {
        std::string kernelDir = rawKernel.substr(0, afterPos).substr(rawKernel.find("/") + 1);
        if (!IsFirstWatchOp(kernelDir)) {
            return EndExcute(stream, kernelDir, rawKernel.substr(0, afterPos), type);
        }
        std::vector<MonitoredTensor> dumpTensors;
        if (outputId_ < tensors.size()) {
            MonitoredTensor tensor {};
            tensor.data = tensors[outputId_].data;
            tensor.dataSize = static_cast<uint64_t>(tensors[outputId_].dataSize);
            dumpTensors.emplace_back(tensor);
            return EndExcute(stream, kernelDir, rawKernel.substr(0, afterPos), type, dumpTensors, outputId_);
        }
        for (auto &item : tensors) {
            MonitoredTensor tensor {};
            tensor.data = item.data;
            tensor.dataSize = static_cast<uint64_t>(item.dataSize);
            dumpTensors.emplace_back(tensor);
        }
        return EndExcute(stream, kernelDir, rawKernel.substr(0, afterPos), type, dumpTensors);
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