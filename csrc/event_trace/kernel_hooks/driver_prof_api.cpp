// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "driver_prof_api.h"
#include "client_process.h"
#include "securec.h"
#include "event_report.h"
#include "runtime_prof_api.h"

namespace Leaks {

static DrvError HalGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    using HalGetDeviceInfoFunc = DrvError(*)(uint32_t, int32_t, int32_t, int64_t*);
    auto vallina = Leaks::VallinaSymbol<DriverProfApiLoader>::Instance().Get<HalGetDeviceInfoFunc>("halGetDeviceInfo");
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("halGetDeviceInfo api get failed");
        return DRV_ERROR_NOT_SUPPORT;
    }

    return vallina(deviceId, moduleType, infoType, value);
}

static int64_t GetDrvVersion(uint32_t deviceId)
{
    constexpr int64_t ERR_VERSION = -1;
    int64_t version = 0;
    DrvError ret = HalGetDeviceInfo(deviceId, DRV_MODULE_TYPE_SYSTEM, DRV_INFO_TYPE_VERSION, &version);
    return (ret == DRV_ERROR_NONE) ? version : ERR_VERSION;
}

static PlatformType GetChipTypeImpl(uint32_t deviceId)
{
    int64_t versionInfo = GetDrvVersion(deviceId);
    if (versionInfo < 0) {
        CLIENT_ERROR_LOG("Call GetDrvVersion failed");
        return PlatformType::END_TYPE;
    }
    uint32_t chipId = ((static_cast<uint64_t>(versionInfo) >> 8) & 0xff);
    if (chipId >= static_cast<uint32_t>(PlatformType::END_TYPE)) {
        CLIENT_ERROR_LOG("Get Chip Type failed");
        return PlatformType::END_TYPE;
    }
    return static_cast<PlatformType>(chipId);
}

static uint64_t GetDevFreq(uint32_t device)
{
    constexpr uint64_t DEFAULT_FREQ = 50;
    static const std::unordered_map<PlatformType, uint64_t> FREQ_MAP = {
        {PlatformType::CHIP_910B, 50},
        {PlatformType::CHIP_310B, 50},
    };
    int64_t freq = 0;
    DrvError ret = HalGetDeviceInfo(device, DRV_MODULE_TYPE_SYSTEM, DRV_INFO_TYPE_DEV_OSC_FREQUE, &freq);
    if (ret != DRV_ERROR_NONE) {
        auto platform = GetChipTypeImpl(device);
        auto iter = FREQ_MAP.find(platform);
        uint64_t defaultFreq = (iter == FREQ_MAP.end()) ? DEFAULT_FREQ : iter->second;
        return defaultFreq;
    }
    return freq;
}

static uint64_t GetClockRealTimeNs()
{
    struct timespec ts;
    if (memset_s(&ts, sizeof(timespec), 0, sizeof(timespec)) != EOK) {
        return 0;
    }
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * SECTONSEC + static_cast<uint64_t>(ts.tv_nsec);
}

static uint64_t GetDevStartSysCnt(uint32_t device)
{
    constexpr uint64_t ERR_SYSCNT = 0;
    int64_t syscnt = 0;
    DrvError ret = HalGetDeviceInfo(device, DRV_MODULE_TYPE_SYSTEM, DRV_INFO_TYPE_SYS_COUNT, &syscnt);
    return (ret == DRV_ERROR_NONE) ? static_cast<uint64_t>(syscnt) : ERR_SYSCNT;
}

DevTimeInfo g_devTimeInfo = { };

static void InitDevTimeInfo(uint32_t deviceId)
{
    static constexpr uint32_t AVE_NUM = 2;

    g_devTimeInfo.freq = GetDevFreq(deviceId);
    auto t1 = GetClockRealTimeNs();
    g_devTimeInfo.startSysCnt = GetDevStartSysCnt(deviceId);
    auto t2 = GetClockRealTimeNs();
    g_devTimeInfo.startRealTime = (t2 + t1) / AVE_NUM;
    return;
}

uint64_t GetRealTimeFromSysCnt(uint32_t deviceId, uint64_t sysCnt)
{
    uint64_t real_time = MSTONS * (sysCnt - g_devTimeInfo.startSysCnt) / g_devTimeInfo.freq +
        g_devTimeInfo.startRealTime;
    return real_time;
}

void StartDriverKernelInfoTrace(int32_t devId)
{
    InitDevTimeInfo(devId);
    SetProfCommand(devId);
    StarsSocLogConfigT configP;
    if (memset_s(&configP, sizeof(StarsSocLogConfigT), 0, sizeof(StarsSocLogConfigT)) != EOK) {
        CLIENT_ERROR_LOG("memset StarsSocLogConfigT failed");
        return;
    }
    configP.acsq_task = TS_PROFILE_COMMAND_TYPE_PROFILING_ENABLE;
    configP.ffts_thread_task = TS_PROFILE_COMMAND_TYPE_PROFILING_ENABLE;
    ProfStartParaT profStartPara;
    profStartPara.channelType = PROF_CHANNEL_TYPE_TS;
    static const uint32_t SAMPLE_PERIOD = 20;
    profStartPara.samplePeriod = SAMPLE_PERIOD;
    profStartPara.realTime = 1;
    profStartPara.userData = &configP;
    profStartPara.userDataSize = static_cast<unsigned int>(sizeof(StarsSocLogConfigT));

    using DriverProfStartFunc = int(*)(unsigned int, unsigned int, struct ProfStartPara*);
    auto vallina = Leaks::VallinaSymbol<DriverProfApiLoader>::Instance().Get<DriverProfStartFunc>("prof_drv_start");
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("DriverProfStartFunc is nullptr");
        return;
    }
    int ret = vallina(static_cast<uint32_t>(devId), PROF_CHANNEL_STARS_SOC_LOG, &profStartPara);
    if (ret != 0) {
        CLIENT_ERROR_LOG("driver prof start failed.");
    }
    return;
}

void EndDriverKernelInfoTrace()
{
    int32_t devId = Leaks::GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == Leaks::GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("get device id failed");
    }
    using DriverProfEndFunc = int(*)(unsigned int, unsigned int);
    auto vallina = Leaks::VallinaSymbol<DriverProfApiLoader>::Instance().Get<DriverProfEndFunc>("prof_stop");
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("EndDriverKernelInfoTrace is nullptr");
        return;
    }

    int ret = vallina(static_cast<uint32_t>(devId), PROF_CHANNEL_STARS_SOC_LOG);
    if (ret != 0) {
        CLIENT_ERROR_LOG("driver prof end failed.");
    }
    return;
}
}