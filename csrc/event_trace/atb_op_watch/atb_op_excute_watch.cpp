// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "atb_op_excute_watch.h"
#include "log.h"

namespace Leaks {

void ATBOpExcuteWatch::BeginExcute(const std::string &rawItem)
{
    if (IsInMonitoring()) {
        DumpATBTensor(rawItem, OpEventType::ATB_START);
        return;
    }
    return;
}

void ATBOpExcuteWatch::EndExcute(const std::string &excuteItem, const std::string &rawItem,
    const std::vector<Tensor> &outputTensors)
{
    if (IsFirstWatchOp(excuteItem)) {
        SetWatchedOpName(excuteItem);
        SetWatchedTensors(outputTensors);

        DumpATBTensor(rawItem, OpEventType::ATEN_END);

        SetMonitoringStatus();
        return;
    }
    if (IsLastWatchOp(excuteItem)) {
        DumpATBTensor(rawItem, OpEventType::ATEN_END);
        
        ClearWatchedOpName();
        ClearWatchedTensors();

        UnSetMonitoringStatus();
        return;
    }
    if (IsInMonitoring()) {
        DumpATBTensor(rawItem, OpEventType::ATEN_END);
        return;
    }

    return;
}

void ATBOpExcuteWatch::AtbOpExcuteBegin(const std::string &rawOp)
{
    std::lock_guard<std::mutex> guard(mutex_);
    return BeginExcute(rawOp);
}

void ATBOpExcuteWatch::AtbOpExcuteEnd(const std::string &rawOp, const atb::SVector<atb::Tensor>& tensors)
{
    std::lock_guard<std::mutex> guard(mutex_);
    auto op = rawOp.substr(rawOp.find("/") + 1);
    std::vector<Tensor> dumpTensors;
    if (!IsFirstWatchOp(op)) {
        return EndExcute(op, rawOp);
    }
    if (outputId_ < tensors.size()) {
        Tensor tensor {};
        tensor.data = tensors[outputId_].deviceData;
        tensor.dataSize = tensors[outputId_].dataSize;
        dumpTensors.emplace_back(tensor);
        return EndExcute(op, rawOp, dumpTensors);
    }
    for (auto &item : tensors) {
        Tensor tensor {};
        tensor.data = item.deviceData;
        tensor.dataSize = item.dataSize;
        dumpTensors.emplace_back(tensor);
    }
    return EndExcute(op, rawOp, dumpTensors);
}

void ATBOpExcuteWatch::AtbKernelExcute(const std::string &rawKernel, const Mki::SVector<Mki::Tensor>& tensors)
{
    std::lock_guard<std::mutex> guard(mutex_);
    auto beforPos = rawKernel.find("/before");
    auto afterPos = rawKernel.find("/after");
    if (beforPos != std::string::npos) {
        return BeginExcute(rawKernel.substr(0, beforPos));
    } else if (afterPos != std::string::npos) {
        std::string kernelDir = rawKernel.substr(0, afterPos).substr(rawKernel.find("/") + 1);
        if (!IsFirstWatchOp(kernelDir)) {
            return EndExcute(kernelDir, rawKernel.substr(0, afterPos));
        }
        std::vector<Tensor> dumpTensors;
        if (outputId_ < tensors.size()) {
            Tensor tensor {};
            tensor.data = tensors[outputId_].data;
            tensor.dataSize = static_cast<uint64_t>(tensors[outputId_].dataSize);
            dumpTensors.emplace_back(tensor);
            return EndExcute(kernelDir, rawKernel.substr(0, afterPos), dumpTensors);
        }
        for (auto &item : tensors) {
            Tensor tensor {};
            tensor.data = item.data;
            tensor.dataSize = static_cast<uint64_t>(item.dataSize);
            dumpTensors.emplace_back(tensor);
        }
        return EndExcute(kernelDir, rawKernel.substr(0, afterPos), dumpTensors);
    } else {
        CLIENT_ERROR_LOG("Invalid kernel info.\n");
        return;
    }
}

void ATBOpExcuteWatch::DumpATBTensor(const std::string &op, OpEventType eventType)
{
    std::vector<Tensor> &tensors = GetWatchedTensors();
    for (size_t i = 0; i < tensors.size(); i++) {
        auto type = (eventType == OpEventType::ATB_START) ? "before" : "after";
        auto fileName = op + "-" + GetWatchedOpName() + "_" + std::to_string(i) + "_" + type + ".bin";
        auto result = ATBTensorDump::GetInstance().Dump(tensors[i], fileName);
        if (!result) {
            LOG_WARN("Dump ATB tensor failed, current op: %s, watched op: %s", op, GetWatchedOpName());
        }
    }
    return;
}

bool ATBOpExcuteWatch::IsFirstWatchOp(const std::string &op)
{
    return op == fistWatchOp_;
}

bool ATBOpExcuteWatch::IsLastWatchOp(const std::string &op)
{
    return op == lastWatchOp_;
}

void ATBOpExcuteWatch::SetWatchedTensors(const std::vector<Tensor> &tensors)
{
    watchedTensors_ = tensors;
}

std::vector<Tensor>& ATBOpExcuteWatch::GetWatchedTensors()
{
    return watchedTensors_;
}

void ATBOpExcuteWatch::ClearWatchedTensors()
{
    watchedTensors_.clear();
}

void ATBOpExcuteWatch::SetWatchedOpName(const std::string &name)
{
    watchedOpName_ = name;
}

std::string ATBOpExcuteWatch::GetWatchedOpName()
{
    return watchedOpName_;
}

void ATBOpExcuteWatch::ClearWatchedOpName()
{
    watchedOpName_ = "";
}

bool ATBOpExcuteWatch::IsInMonitoring()
{
    return inMonitoring_;
}

void ATBOpExcuteWatch::SetMonitoringStatus()
{
    inMonitoring_ = true;
}

void ATBOpExcuteWatch::UnSetMonitoringStatus()
{
    inMonitoring_ = false;
}

}