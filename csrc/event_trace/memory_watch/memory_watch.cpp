// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "memory_watch.h"

namespace Leaks {

uint64_t MemoryWatch::CountOpName(const std::string& name)
{
    if (targetNameCnt_.find(name) == targetNameCnt_.end()) {
        targetNameCnt_[name] = 1;
    } else {
        targetNameCnt_[name] += 1;
    }
    return targetNameCnt_[name];
}

void MemoryWatch::BeginExcute(aclrtStream stream, const std::string &rawItem)
{
    auto name = rawItem + "_" + std::to_string(CountOpName(rawItem.substr(rawItem.find("/") + 1)));
    if (TensorMonitor::GetInstance().IsInMonitoring()) {
        TensorDumper::GetInstance().Dump(stream, name, true);
        return;
    }
    return;
}

void MemoryWatch::EndExcute(aclrtStream stream, const std::string &excuteItem, const std::string &rawItem,
    const std::vector<MonitoredTensor> &outputTensors,  uint32_t outputId)
{
    std::string name;
    if (IsFirstWatchTarget(excuteItem) && watchedTargetName_.empty()) {
        SetWatchedTargetName(excuteItem);
        name = rawItem + "_" + std::to_string(firstWatchTargetCnt_++);
        TensorMonitor::GetInstance().AddWatchTensor(outputTensors, outputId);
        TensorDumper::GetInstance().Dump(stream, name, false);

        return;
    }
    name = rawItem + "_" + std::to_string(targetNameCnt_[excuteItem]);
    if (IsLastWatchTarget(excuteItem)) {
        TensorDumper::GetInstance().Dump(stream, name, false);
        
        ClearWatchedTargetName();
        TensorMonitor::GetInstance().ClearCmdWatchTensor();

        return;
    }
    if (TensorMonitor::GetInstance().IsInMonitoring()) {
        TensorDumper::GetInstance().Dump(stream, name, false);
        return;
    }

    return;
}

void OpExcuteBegin(aclrtStream stream, char *rawOp)
{
    std::string str(rawOp);
    return Leaks::MemoryWatch::GetInstance().OpExcuteBegin(stream, str);
}

void OpExcuteEnd(aclrtStream stream, char *rawOp, MonitoredTensor* tensorsInput, size_t size)
{
    std::vector<MonitoredTensor> tensors;

    tensors.reserve(size);

    for (size_t i = 0; i < size; ++i) {
        tensors.push_back(tensorsInput[i]);
    }
    std::string str(rawOp);
    return Leaks::MemoryWatch::GetInstance().OpExcuteEnd(stream, str, tensors);
}

void MemoryWatch::OpExcuteBegin(aclrtStream stream, const std::string &rawOp)
{
    std::lock_guard<std::mutex> guard(mutex_);
    return BeginExcute(stream, rawOp);
}

void MemoryWatch::OpExcuteEnd(aclrtStream stream,
    const std::string &rawOp, const std::vector<MonitoredTensor>& tensors)
{
    std::lock_guard<std::mutex> guard(mutex_);
    auto op = rawOp.substr(rawOp.find("/") + 1);
    if (!IsFirstWatchTarget(op)) {
        return EndExcute(stream, op, rawOp);
    }
    if (outputId_ < tensors.size()) {
        std::vector<MonitoredTensor> dumpTensors;
        MonitoredTensor tensor = tensors[outputId_];
        dumpTensors.emplace_back(tensor);
        return EndExcute(stream, op, rawOp, dumpTensors, outputId_);
    } else {
        outputId_ = UINT32_MAX;
    }
    return EndExcute(stream, op, rawOp, tensors);
}

void MemoryWatch::KernelExcuteBegin(aclrtStream stream, const std::string &rawKernel, bool isOuterLayer)
{
    std::lock_guard<std::mutex> guard(mutex_);
    // 防止atb的kernel监控与python接口的kernel监控重复
    if (isOuterLayer) {
        isRepeatWatch_[Utility::GetTid()] = true;
    }
    if (isRepeatWatch_[Utility::GetTid()] && !isOuterLayer) {
        return ;
    }
    BeginExcute(stream, rawKernel);
}

void MemoryWatch::KernelExcuteEnd(aclrtStream stream, const std::string &rawKernel, bool isOuterLayer,
    const Mki::SVector<Mki::Tensor>& tensors)
{
    std::lock_guard<std::mutex> guard(mutex_);
    // 防止atb的kernel监控与python接口的kernel监控重复
    if (isRepeatWatch_[Utility::GetTid()] && !isOuterLayer) {
        return ;
    }
    if (isOuterLayer) {
        isRepeatWatch_[Utility::GetTid()] = false;
    }
    std::string kernelDir = rawKernel.substr(rawKernel.find("/") + 1);
    if (!IsFirstWatchTarget(kernelDir)) {
        return EndExcute(stream, kernelDir, rawKernel);
    }
    std::vector<MonitoredTensor> dumpTensors;
    if (outputId_ < tensors.size()) {
        MonitoredTensor tensor {};
        tensor.data = tensors[outputId_].data;
        tensor.dataSize = static_cast<uint64_t>(tensors[outputId_].dataSize);
        dumpTensors.emplace_back(tensor);
        return EndExcute(stream, kernelDir, rawKernel, dumpTensors, outputId_);
    } else {
        outputId_ = UINT32_MAX;
    }
    for (auto &item : tensors) {
        MonitoredTensor tensor {};
        tensor.data = item.data;
        tensor.dataSize = static_cast<uint64_t>(item.dataSize);
        dumpTensors.emplace_back(tensor);
    }
    EndExcute(stream, kernelDir, rawKernel, dumpTensors, outputId_);
}

void ATBKernelExcute(aclrtStream stream, char* rawKernel, const Mki::SVector<Mki::Tensor>& tensors)
{
    std::string str(rawKernel);
    Leaks::MemoryWatch::GetInstance().ATBKernelExcute(stream, str, tensors);
}

void MemoryWatch::ATBKernelExcute(aclrtStream stream, std::string rawKernel, const Mki::SVector<Mki::Tensor>& tensors)
{
    auto beforPos = rawKernel.find("/before");
    auto afterPos = rawKernel.find("/after");
    if (beforPos != std::string::npos) {
        KernelExcuteBegin(stream, rawKernel.substr(0, beforPos), true);
    } else if (afterPos != std::string::npos) {
        KernelExcuteEnd(stream, rawKernel.substr(0, afterPos), true, tensors);
    } else {
        LOG_ERROR("Invalid kernel info.\n");
        return;
    }
}

bool MemoryWatch::IsFirstWatchTarget(const std::string &name)
{
    return name == firstWatchTarget_;
}

bool MemoryWatch::IsLastWatchTarget(const std::string &name)
{
    return name == lastWatchTarget_;
}

void MemoryWatch::SetWatchedTargetName(const std::string &name)
{
    watchedTargetName_ = name;
}

std::string MemoryWatch::GetWatchedTargetName()
{
    return watchedTargetName_;
}

void MemoryWatch::ClearWatchedTargetName()
{
    watchedTargetName_ = "";
}

}