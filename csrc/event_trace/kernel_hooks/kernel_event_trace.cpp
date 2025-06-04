// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "kernel_event_trace.h"
#include "event_report.h"
#include "client_process.h"
#include "utils.h"
#include "driver_prof_api.h"
#include "stars_common.h"
#include "securec.h"
#include "bit_field.h"

namespace Leaks {

void KernelEventTrace::KernelLaunch(const AclnnKernelMapInfo &kernelLaunchInfo)
{
    if (!EventReport::Instance(CommType::SOCKET).ReportKernelLaunch(kernelLaunchInfo)) {
        CLIENT_ERROR_LOG("KernelLaunch launch report failed");
    }
    
    return;
}

void KernelEventTrace::KernelStartExcute(const TaskKey& key, uint64_t time)
{
    auto kernelName = RuntimeKernelLinker::GetInstance().GetKernelName(key, KernelEventType::KERNEL_START);
    if (!kernelName.empty()) {
        if (!EventReport::Instance(CommType::SOCKET).ReportKernelExcute(key,
            kernelName, time, KernelEventType::KERNEL_START)) {
            CLIENT_ERROR_LOG("Kernel excute start report failed");
        }
    }
    return;
}

void KernelEventTrace::KernelEndExcute(const TaskKey& key, uint64_t time)
{
    auto kernelName = RuntimeKernelLinker::GetInstance().GetKernelName(key, KernelEventType::KERNEL_END);
    if (!kernelName.empty()) {
        if (!EventReport::Instance(CommType::SOCKET).ReportKernelExcute(key,
            kernelName, time, KernelEventType::KERNEL_END)) {
            CLIENT_ERROR_LOG("Kernel excute end report failed");
        }
    }
    return;
}

static void ReportStarsSocLog(uint32_t deviceId, const StarsSocLog* socLog)
{
    if (!socLog) {
        return;
    }
    constexpr int32_t bitOffset = 32;
    uint16_t streamId = GetStreamId(static_cast<uint16_t>(socLog->streamId), static_cast<uint16_t>(socLog->taskId));
    uint16_t taskId = GetTaskId(static_cast<uint16_t>(socLog->streamId), static_cast<uint16_t>(socLog->taskId));
    auto taskKey = std::make_tuple(static_cast<uint16_t>(deviceId), streamId, taskId);
    if (socLog->funcType == STARS_FUNC_TYPE_BEGIN) {
        auto start = static_cast<uint64_t>(socLog->sysCntH) << bitOffset | socLog->sysCntL;
        start = GetRealTimeFromSysCnt(deviceId, start);
        KernelEventTrace::GetInstance().KernelStartExcute(taskKey, start);
    } else if (socLog->funcType == STARS_FUNC_TYPE_END) {
        auto end = static_cast<uint64_t>(socLog->sysCntH) << bitOffset | socLog->sysCntL;
        end = GetRealTimeFromSysCnt(deviceId, end);
        KernelEventTrace::GetInstance().KernelEndExcute(taskKey, end);
    }

    return ;
}

static size_t TransStarsLog(char buffer[], size_t validSize, uint32_t deviceId)
{
    size_t pos = 0;
    while (validSize - pos >= sizeof(StarsSocLog)) {
        StarsSocLog* data = reinterpret_cast<StarsSocLog*>(buffer + pos);
        ReportStarsSocLog(deviceId, data);
        pos += sizeof(StarsSocLog);
    }
    return pos;
}

static size_t TransDataToActivityBuffer(char buffer[], size_t validSize,
    uint32_t deviceId, AI_DRV_CHANNEL channelId)
{
    switch (channelId) {
        case PROF_CHANNEL_STARS_SOC_LOG:
            return TransStarsLog(buffer, validSize, deviceId);
        default:
            return 0;
    }
}

void KernelEventTrace::CreateReadDataChannel(uint32_t devId)
{
    readTh_ = std::thread([devId, this]()mutable {
        char buf[MAX_BUFFER_SIZE] = {0};
        size_t curPos = 0;
        int currLen = 0;
        using ReadFunc = int(*)(unsigned int, unsigned int, char*, unsigned int);
        while (started_) {
            static auto vallina = VallinaSymbol<DriverProfApiLoader>::Instance().Get<ReadFunc>("prof_channel_read");
            if (vallina == nullptr) {
                CLIENT_ERROR_LOG("ReadFunc is null");
                return;
            }
            currLen = vallina(devId, PROF_CHANNEL_STARS_SOC_LOG, buf + curPos, MAX_BUFFER_SIZE - curPos);
            if (currLen <= 0) {
                continue;
            }
            auto uintCurrLen = static_cast<size_t>(currLen);
            if (uintCurrLen >= (MAX_BUFFER_SIZE - curPos)) {
                CLIENT_ERROR_LOG("Read invalid data len from driver");
                continue;
            }
            size_t lastPos = TransDataToActivityBuffer(buf, curPos + uintCurrLen,
                devId, PROF_CHANNEL_STARS_SOC_LOG);
            if (lastPos < curPos + uintCurrLen) {
                if (memcpy_s(buf, MAX_BUFFER_SIZE, buf + lastPos, curPos + uintCurrLen - lastPos) != EOK) {
                    continue;
                }
            }
            curPos = curPos + uintCurrLen - lastPos;
        }
    });
}

void KernelEventTrace::StartKernelEventTrace()
{
    Config config = EventReport::Instance(CommType::SOCKET).GetConfig();
    BitField<decltype(config.levelType)> levelType(config.levelType);
    if (!levelType.checkBit(static_cast<size_t>(LevelType::LEVEL_KERNEL))) {
        return;
    }
    BitField<decltype(config.eventType)> eventType(config.eventType);
    if (!eventType.checkBit(static_cast<size_t>(EventType::LAUNCH_EVENT))) {
        return;
    }
    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("get device id failed");
    }

    StartDriverKernelInfoTrace(devId);
    CreateReadDataChannel(static_cast<uint32_t>(devId));
}

void KernelEventTrace::EndKernelEventTrace()
{
    return EndDriverKernelInfoTrace();
}

KernelEventTrace::~KernelEventTrace()
{
    started_.store(false);
    if (readTh_.joinable()) {
        readTh_.join();
    }
}

void RuntimeKernelLinker::RuntimeTaskInfoLaunch(const TaskKey& key, uint64_t hashId)
{
    if (hashId == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = kernelNameMp_.find(Utility::GetTid());
    if (iter == kernelNameMp_.end()) {
        return;
    }
    auto &vec = iter->second;
    if (!vec.empty()) {
        vec.back().taskKey = key;
        vec.back().kernelName = GetHashInfo(hashId);
        KernelEventTrace::GetInstance().KernelLaunch(vec.back());
    }

    return;
}

void RuntimeKernelLinker::KernelLaunch()
{
    uint64_t timeStamp = Utility::GetTimeNanoseconds();
    std::lock_guard<std::mutex> lock(mutex_);

    AclnnKernelMapInfo value = {timeStamp, std::make_tuple(-1, -1, -1), ""};
    auto iter = kernelNameMp_.find(Utility::GetTid());
    if (iter == kernelNameMp_.end()) {
        std::vector<AclnnKernelMapInfo> vec {};
        vec.push_back(value);
        kernelNameMp_.insert({Utility::GetTid(), vec});
    } else {
        auto &vec = iter->second;
        vec.push_back(value);
    }

    return;
}

std::string RuntimeKernelLinker::GetKernelName(const TaskKey& key, KernelEventType type)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &pair : kernelNameMp_) {
        auto& vec = pair.second;
        for (auto it = vec.begin(); it != vec.end(); ++it) {
            if (it->taskKey == key) {
                std::string name = it->kernelName;
                if (type == KernelEventType::KERNEL_END) {
                    vec.erase(it); // 使用完删除
                }
                return name;
            }
        }
    }
    return "";
}

void RuntimeKernelLinker::SetHashInfo(uint64_t hashId, const std::string &hashInfo)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = hashInfo_map_.find(hashId);
    if (iter == hashInfo_map_.end()) {
        hashInfo_map_.insert({hashId, hashInfo});
    }
}

std::string RuntimeKernelLinker::GetHashInfo(uint64_t hashId)
{
    const auto iter = hashInfo_map_.find(hashId);
    if (iter != hashInfo_map_.end()) {
        return iter->second;
    }
    return "";
}
}