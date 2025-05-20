// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "aten_manager.h"
#include <cstring>
#include "securec.h"
#include "call_stack.h"
#include "ustring.h"
#include "log.h"


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

void AtenManager::ReportAtenAccess(const char* msg, int32_t streamId)
{
    MemAccessRecord record;

    std::string addr;
    std::string dtype;
    std::string shape;
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

    if (isWrite == "False" && isRead == "False") {
        record.eventType = AccessType::UNKNOWN;
    } else if (isWrite == "True") {
        record.eventType = AccessType::WRITE;
    } else {
        record.eventType = AccessType::READ;
    }
    record.memType = AccessMemType::ATEN;

    if (!Utility::StrToUint64(record.addr, addr)) {
        CLIENT_ERROR_LOG("Aten Tensor's addr StrToUint64 failed");
    }
    if (!Utility::StrToUint64(record.memSize, size)) {
        CLIENT_ERROR_LOG("Aten Tensor's memSize StrToUint64 failed");
    }
    // 组装attr属性
    std::ostringstream oss;
    oss << "dtype:" << dtype << ",shape:" << shape;
    std::string attr = oss.str();
    strncpy_s(record.attr, sizeof(record.attr), attr.c_str(), sizeof(record.attr) - 1);
    if (strncpy_s(record.name, sizeof(record.name), name.c_str(), sizeof(record.name) - 1) != EOK) {
        CLIENT_ERROR_LOG("strncpy_s FAILED");
        record.name[0] = '\0';
    }

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