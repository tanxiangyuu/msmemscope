// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "event_trace_manager.h"
#include "event_report.h"
#include "cpython.h"
#include "bit_field.h"
#include "client_parser.h"

namespace Leaks {

const std::unordered_map<std::string, std::function<void(const std::string&, Config&, bool&)>> parserConfigTable = {
    {"--call-stack", ParseCallstack},
    {"--level", ParseDataLevel},
    {"--events", ParseEventTraceType},
    {"--device", ParseDevice},
};

ConfigManager::ConfigManager()
{
    config_ = EventReport::Instance(CommType::SOCKET).GetInitConfig();
}

Config ConfigManager::GetConfig()
{
    return config_;
}

void ConfigManager::SetConfigImpl(const Config &config)
{
    std::lock_guard<std::mutex> lock(mutex_);

    config_ = config;

    SetEventDefaultConfig(config_);
    SetAnalysisDefaultConfig(config_);

    g_isReportHostMem = config_.collectCpu;
    EventTraceManager::Instance().HandleWithATenCollect();
}

void ConfigManager::SetConfig(const Config &config)
{
    SetConfigImpl(config);
}

bool ConfigManager::SetConfig(const std::unordered_map<std::string, std::string> &pythonConfig)
{
    Config config = config_;
    for (auto &p : pythonConfig) {
        auto itr = parserConfigTable.find(p.first);
        if (itr == parserConfigTable.end()) {
            return false;
        }
        bool parseFail = false;
        itr->second(p.second, config, parseFail);
        if (parseFail) {
            return false;
        }
    }

    SetConfigImpl(config);

    return true;
}

bool EventTraceManager::IsNeedTraceOp()
{
    return BitPresent(GetConfig().levelType, static_cast<size_t>(LevelType::LEVEL_OP));
}

bool EventTraceManager::IsNeedTraceKernel()
{
    return BitPresent(GetConfig().levelType, static_cast<size_t>(LevelType::LEVEL_KERNEL));
}

bool EventTraceManager::IsNeedTraceAccess()
{
    return BitPresent(GetConfig().eventType, static_cast<size_t>(EventType::ACCESS_EVENT));
}

bool EventTraceManager::IsNeedTraceLaunch()
{
    return BitPresent(GetConfig().eventType, static_cast<size_t>(EventType::LAUNCH_EVENT));
}

bool EventTraceManager::IsNeedTraceAlloc()
{
    return BitPresent(GetConfig().eventType, static_cast<size_t>(EventType::ALLOC_EVENT));
}

bool EventTraceManager::IsNeedTraceFree()
{
    return BitPresent(GetConfig().eventType, static_cast<size_t>(EventType::FREE_EVENT));
}

bool EventTraceManager::IsNeedTraceKernelLaunch()
{
    return IsNeedTraceKernel() && IsNeedTraceLaunch();
}

bool EventTraceManager::IsNeedTraceOpLaunch()
{
    return IsNeedTraceOp() && IsNeedTraceLaunch();
}

bool EventTraceManager::IsNeedTraceMemory()
{
    return IsNeedTraceAlloc() && IsNeedTraceFree();
}

void EventTraceManager::InitJudgeFuncTable()
{
    jdugeFuncTable_ = {
        {RecordType::KERNEL_LAUNCH_RECORD, [this]() { return IsNeedTraceKernelLaunch(); }},
        {RecordType::KERNEL_EXCUTE_RECORD, [this]() { return IsNeedTraceKernelLaunch(); }},
        {RecordType::MEMORY_POOL_RECORD, [this]() { return IsNeedTraceMemory(); }},
        {RecordType::MEMORY_RECORD, [this]() { return IsNeedTraceMemory(); }},
        {RecordType::ATB_OP_EXECUTE_RECORD, [this]() { return IsNeedTraceOp(); }},
        {RecordType::ATEN_OP_LAUNCH_RECORD, [this]() { return IsNeedTraceOpLaunch(); }},
        {RecordType::ATB_KERNEL_RECORD, [this]() { return IsNeedTraceKernel(); }},
        {RecordType::MEM_ACCESS_RECORD, [this]() { return IsNeedTraceAccess(); }},
        {RecordType::OP_LAUNCH_RECORD, [this]() { return IsNeedTraceOpLaunch(); }},
    };
}

// 1、判断是否处在采集范围
// 2、判断当前的采集项是否需要采集
bool EventTraceManager::IsNeedTrace(const RecordType type)
{
    if (status_ != EventTraceStatus::IN_TRACING) {
        return false;
    }

    auto itr = jdugeFuncTable_.find(type);
    if (itr == jdugeFuncTable_.end()) {
        return true;
    }

    return itr->second();
}

void EventTraceManager::InitTraceStatus()
{
    auto status = (GetConfig().collectMode == static_cast<uint8_t>(CollectMode::FULL)) ? EventTraceStatus::IN_TRACING :
        EventTraceStatus::NOT_IN_TRACING;
    SetTraceStatus(status);
    return;
}

void EventTraceManager::SetTraceStatus(const EventTraceStatus status)
{
    CLIENT_INFO_LOG("Set trace status to " + std::to_string(static_cast<uint8_t>(status)) + " .");

    if (!EventReport::Instance(CommType::SOCKET).ReportTraceStatus(status)) {
        CLIENT_ERROR_LOG("Report trace status failed.\n");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    status_ = status;

    HandleWithATenCollect();
    return;
}

void EventTraceManager::HandleWithATenCollect()
{
    if ((status_ == EventTraceStatus::IN_TRACING) && IsNeedTraceOp() && aclInit_) {
        Utility::LeaksPythonCall("msleaks.aten_collection", "enable_aten_collector");
        return;
    }

    Utility::LeaksPythonCall("msleaks.aten_collection", "disable_aten_collector");

    return;
}

void EventTraceManager::SetAclInitStatus(bool isInit)
{
    aclInit_ = isInit;

    HandleWithATenCollect();
}

}