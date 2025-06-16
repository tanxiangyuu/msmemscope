// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "aten_manager.h"
#include <cstring>
#include "securec.h"
#include "call_stack.h"
#include "ustring.h"
#include "log.h"

#include <iostream>


namespace Leaks {
    
AtenManager& AtenManager::GetInstance()
{
    static AtenManager instance;
    return instance;
}

bool AtenManager::ExtractTensorInfo(const char* msg, const std::string &key, std::string &value)
{
    std::string msgString(msg);
    size_t startPos = msgString.find(key);
    if (startPos == std::string::npos) {
        return false;
    }
    startPos += key.length();
    size_t endPos = msgString.find_first_of(";}", startPos);
    if (endPos == std::string::npos) {
        endPos = msgString.length();
    }
    value = msgString.substr(startPos, endPos - startPos);
    return true;
}

bool AtenManager::IsAtenLaunchEnable()
{
    // 命令行判断是否包含launch事件
    Config userConfig =  EventReport::Instance(CommType::SOCKET).GetConfig();
    BitField<decltype(userConfig.eventType)> eventType(userConfig.eventType);
    if (eventType.checkBit(static_cast<size_t>(EventType::LAUNCH_EVENT))) {
        return true;
    }
    return false;
}

bool AtenManager::IsAtenAccessEnable()
{
    // 命令行判断是否包含Access事件
    Config userConfig =  EventReport::Instance(CommType::SOCKET).GetConfig();
    BitField<decltype(userConfig.eventType)> eventType(userConfig.eventType);
    if (eventType.checkBit(static_cast<size_t>(EventType::ACCESS_EVENT))) {
        return true;
    }
    return false;
}

bool AtenManager::IsWatchEnable()
{
    Config userConfig =  EventReport::Instance(CommType::SOCKET).GetConfig();
    return userConfig.watchConfig.isWatched;
}

void AtenManager::ProcessMsg(const char* msg, int32_t streamId)
{
    // 根据标识判断是否为aten算子下发或者tensor信息
    bool isAtenBegin;
    if (strncmp(msg, ATEN_BEGIN_MSG, strlen(ATEN_BEGIN_MSG)) == 0) {
        isAtenBegin = true;
        ReportAtenLaunch(msg, streamId, isAtenBegin);
        return;
    }
    if (strncmp(msg, ATEN_END_MSG, strlen(ATEN_END_MSG)) == 0) {
        isAtenBegin = false;
        ReportAtenLaunch(msg, streamId, isAtenBegin);
        return;
    }
    if (strncmp(msg, ACCESS_MSG, strlen(ACCESS_MSG)) == 0) {
        ReportAtenAccess(msg, streamId);
        return;
    }
}

void AtenManager::ReportAtenLaunch(const char* msg, int32_t streamId, bool isAtenBegin)
{
    AtenOpLaunchRecord record;
    if (isAtenBegin) {
        record.eventType = OpEventType::ATEN_START;
    } else {
        record.eventType = OpEventType::ATEN_END;
    }

    const char* eventName;
    const char* lastSpace = strrchr(msg, ' ');
    if (lastSpace != nullptr) {
        eventName = lastSpace + 1;
    } else {
        eventName = "N/A";
    }
    strncpy_s(record.name, sizeof(record.name), eventName, sizeof(record.name) - 1);

    if (IsWatchEnable() && isAtenBegin) {
        OpExcuteWatch::GetInstance().OpExcuteBegin(nullptr, std::string(eventName), OpType::ATEN);
    }
    if (IsWatchEnable() && !isAtenBegin) {
        OpExcuteWatch::GetInstance().OpExcuteEnd(nullptr, std::string(eventName), outputTensors_, OpType::ATEN);
        outputTensors_.clear();
    }

    if (!IsAtenLaunchEnable()) {
        return ;
    }

    auto config = EventReport::Instance(CommType::SOCKET).GetConfig();
    std::string cStack;
    std::string pyStack;
    if (config.enablePyStack) {
        Utility::GetPythonCallstack(config.pyStackDepth, pyStack);
    }
    CallStackString stack{cStack, pyStack};
    if (!EventReport::Instance(CommType::SOCKET).ReportAtenLaunch(record, stack)) {
        CLIENT_ERROR_LOG("Report Aten Launch FAILED");
    }
    return;
}

void AtenManager::ParseAtenAccessMsg(const char* msg, MemAccessRecord &record, std::string &dtype,
    std::string &shape, std::string &isOutput)
{
    std::string addr;
    std::string size;
    std::string name;
    std::string isRead;
    std::string isWrite;
    ExtractTensorInfo(msg, "ptr=", addr);
    ExtractTensorInfo(msg, "dtype=", dtype);
    ExtractTensorInfo(msg, "shape=", shape);
    ExtractTensorInfo(msg, "tensor_size=", size);
    ExtractTensorInfo(msg, "name=", name);
    ExtractTensorInfo(msg, "is_write=", isWrite);
    ExtractTensorInfo(msg, "is_read=", isRead);
    ExtractTensorInfo(msg, "is_output=", isOutput);

    if (isWrite == "False" && isRead == "False") {
        record.eventType = AccessType::UNKNOWN;
    } else if (isWrite == "True") {
        record.eventType = AccessType::WRITE;
    } else {
        record.eventType = AccessType::READ;
    }
    record.memType = OpType::ATEN;

    if (!Utility::StrToUint64(record.addr, addr)) {
        CLIENT_ERROR_LOG("Aten Tensor's addr StrToUint64 failed");
    }
    if (!Utility::StrToUint64(record.memSize, size)) {
        CLIENT_ERROR_LOG("Aten Tensor's memSize StrToUint64 failed");
    }
    if (strncpy_s(record.name, sizeof(record.name), name.c_str(), sizeof(record.name) - 1) != EOK) {
        CLIENT_ERROR_LOG("strncpy_s FAILED");
        record.name[0] = '\0';
    }
}

void AtenManager::ReportAtenAccess(const char* msg, int32_t streamId)
{
    MemAccessRecord record;
    std::string dtype;
    std::string shape;
    std::string isOutput;
    ParseAtenAccessMsg(msg, record, dtype, shape, isOutput);

    if (isOutput == "True" && IsWatchEnable()) {
        MonitoredTensor tensorInfo{};
        tensorInfo.data =  reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(record.addr));
        tensorInfo.dataSize = record.memSize;
        outputTensors_.push_back(tensorInfo);
    }

    if (!IsAtenAccessEnable()) {
        return ;
    }

    // 组装attr属性
    std::ostringstream oss;
    oss << "dtype:" << dtype << ",shape:" << shape;
    std::string attr = oss.str();
    strncpy_s(record.attr, sizeof(record.attr), attr.c_str(), sizeof(record.attr) - 1);

    auto config = EventReport::Instance(CommType::SOCKET).GetConfig();
    std::string cStack;
    std::string pyStack;
    if (config.enablePyStack) {
        Utility::GetPythonCallstack(config.pyStackDepth, pyStack);
    }
    CallStackString stack{cStack, pyStack};
    if (!EventReport::Instance(CommType::SOCKET).ReportAtenAccess(record, stack)) {
        CLIENT_ERROR_LOG("Report Aten Access FAILED");
    }
    return;
}

}