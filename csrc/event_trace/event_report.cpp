// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "event_report.h"
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include "log.h"
#include "protocol.h"
#include "serializer.h"
#include "utils.h"
#include "vallina_symbol.h"
#include "ustring.h"
#include "umask_guard.h"
#include "securec.h"
#include "bit_field.h"
#include "kernel_hooks/runtime_prof_api.h"
#include "describe_trace.h"

namespace Leaks {
thread_local bool g_isReportHostMem = false;
thread_local bool g_isInReportFunction = false;

constexpr uint64_t MEM_MODULE_ID_BIT = 56;
constexpr uint64_t MEM_VIRT_BIT = 10;
constexpr uint64_t MEM_SVM_VAL = 0x0;
constexpr uint64_t MEM_DEV_VAL = 0x1;
constexpr uint64_t MEM_HOST_VAL = 0x2;
constexpr uint64_t MEM_DVPP_VAL = 0x3;
constexpr uint32_t MAX_THREAD_NUM = 200;

MemOpSpace GetMemOpSpace(unsigned long long flag)
{
    // bit10~13: virt mem type(svm\dev\host\dvpp)
    int32_t memType = (flag & 0b11110000000000) >> MEM_VIRT_BIT;
    MemOpSpace space = MemOpSpace::INVALID;
    switch (memType) {
        case MEM_SVM_VAL:
            space = MemOpSpace::SVM;
            break;
        case MEM_DEV_VAL:
            space = MemOpSpace::DEVICE;
            break;
        case MEM_HOST_VAL:
            space = MemOpSpace::HOST;
            break;
        case MEM_DVPP_VAL:
            space = MemOpSpace::DVPP;
            break;
        default:
            CLIENT_ERROR_LOG("No matching memType for " + std::to_string(memType));
    }
    return space;
}
inline int32_t GetMallocModuleId(unsigned long long flag)
{
    // bit56~63: model id
    return (flag & 0xFF00000000000000) >> MEM_MODULE_ID_BIT;
}

constexpr unsigned long long FLAG_INVALID = UINT64_MAX;

constexpr int32_t INVALID_MODID = -1;

RTS_API rtError_t GetDevice(int32_t *devId)
{
    char const *sym = "rtGetDevice";
    using RtGetDevice = decltype(&GetDevice);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtGetDevice>(sym);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }
    rtError_t ret = vallina(devId);
    return ret;
}

MemOpRecord CreateMemRecord(MemOpType type, unsigned long long flag, MemOpSpace space, uint64_t addr, uint64_t size)
{
    MemOpRecord record;
    record.timeStamp = Utility::GetTimeMicroseconds();
    record.flag = flag;
    record.memType = type;
    record.space = space;
    record.addr = addr;
    record.memSize = size;
    record.pid = Utility::GetPid();
    record.tid = Utility::GetTid();
    return record;
}

AclItfRecord CreateAclItfRecord(AclOpType type)
{
    auto record = AclItfRecord {};
    record.timeStamp = Utility::GetTimeMicroseconds();
    record.type = type;
    record.pid = Utility::GetPid();
    record.tid = Utility::GetTid();
    return record;
}

int EventReport::ReportRecordEvent(EventRecord &record, PacketHead &head, CallStackString& stack)
{
    record.cStackLen = stack.cStack.size();
    record.pyStackLen = stack.pyStack.size();
    std::string buffer = Serialize<PacketHead, EventRecord>(head, record) + stack.cStack + stack.pyStack;
    auto sendNums = ClientProcess::GetInstance(CommType::SOCKET).Notify(buffer);
    return sendNums;
}

int EventReport::ReportRecordEvent(EventRecord &record, PacketHead &head)
{
    std::string buffer = Serialize<PacketHead, EventRecord>(head, record);
    auto sendNums = ClientProcess::GetInstance(CommType::SOCKET).Notify(buffer);
    return sendNums;
}

EventReport& EventReport::Instance(CommType type)
{
    static EventReport instance(type);
    return instance;
}

void EventReport::Init()
{
    recordIndex_.store(0);
    kernelLaunchRecordIndex_.store(0);
    aclItfRecordIndex_.store(0);
    isReceiveServerInfo_.store(false);
}

Config EventReport::GetConfig()
{
    return config_;
}

EventReport::EventReport(CommType type)
{
    Init();

    (void)ClientProcess::GetInstance(type); // 连接server
    std::string msg;
    uint32_t reTryTimes = 5; // 当前系统设置（setsockopt）的read超时时长是1s，默认至多尝试5次
    isReceiveServerInfo_ = (ClientProcess::GetInstance(type).Wait(msg, reTryTimes) > 0) ? true : false;
    Deserialize(msg, config_);

    BitField<decltype(config_.levelType)> levelType(config_.levelType);
    if (levelType.checkBit(static_cast<size_t>(LevelType::LEVEL_KERNEL))) {
        RegisterRtProfileCallback();
    }
    return;
}

bool EventReport::IsNeedSkip()
{
    auto stepList = config_.stepList;

    if (stepList.stepCount == 0) {
        return false;
    }

    for (uint8_t loop = 0; (loop < stepList.stepCount && loop < SELECTED_STEP_MAX_NUM); loop++) {
        if (stepInfo_.currentStepId == stepList.stepIdList[loop] && stepInfo_.inStepRange) {
            return false;
        }
    }

    return true;
}

bool EventReport::IsConnectToServer()
{
    return isReceiveServerInfo_;
}

void GetOwner(char *res, uint32_t size)
{
    std::string owner = DescribeTrace::GetInstance().GetDescribe();
    if (strncpy_s(res, size, owner.c_str(), size - 1) != EOK) {
        CLIENT_ERROR_LOG("strncpy_s FAILED");
    }
}

bool EventReport::ReportAddrInfo(AddrInfo &info)
{
    g_isInReportFunction = true;
    if (!IsConnectToServer()) {
        return true;
    }
    PacketHead head = {PacketType::RECORD};
    EventRecord eventRecord;
    eventRecord.type = RecordType::ADDR_INFO_RECORD;
    eventRecord.record.addrInfo = info;
    CallStackString stack;
    auto sendNums = ReportRecordEvent(eventRecord, head, stack);
    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportMemPoolRecord(MemPoolRecord &memPoolRecord, CallStackString& stack)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    if (IsNeedSkip()) {
        return true;
    }

    BitField<decltype(config_.eventType)> eventType(config_.eventType);
    // 根据命令行参数判断malloc和free是否上报, 0为malloc，剩下的为free
    if (memPoolRecord.memoryUsage.dataType == 0) {
        if (!eventType.checkBit(static_cast<size_t>(EventType::ALLOC_EVENT))) {
            return true;
        }
    } else {
        if (!eventType.checkBit(static_cast<size_t>(EventType::FREE_EVENT))) {
            return true;
        }
    }

    PacketHead head = {PacketType::RECORD};
    EventRecord eventRecord;
    eventRecord.type = memPoolRecord.type;
    eventRecord.record.memPoolRecord = memPoolRecord;
    eventRecord.record.memPoolRecord.timeStamp = Utility::GetTimeMicroseconds();
    eventRecord.record.memPoolRecord.kernelIndex = kernelLaunchRecordIndex_;
    eventRecord.record.memPoolRecord.devId = static_cast<int32_t>(memPoolRecord.memoryUsage.deviceIndex);
    if (eventRecord.record.memPoolRecord.memoryUsage.dataType) {
        eventRecord.record.memPoolRecord.owner[0] = '\0';
    } else {
        GetOwner(eventRecord.record.memPoolRecord.owner, sizeof(eventRecord.record.memPoolRecord.owner));
    }
    eventRecord.record.memPoolRecord.recordIndex = ++recordIndex_;
    auto sendNums = ReportRecordEvent(eventRecord, head, stack);

    g_isInReportFunction = false;

    return (sendNums >= 0);
}

bool EventReport::ReportMalloc(uint64_t addr, uint64_t size, unsigned long long flag, CallStackString& stack)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    if (IsNeedSkip()) {
        return true;
    }

    BitField<decltype(config_.eventType)> eventType(config_.eventType);
    if (!eventType.checkBit(static_cast<size_t>(EventType::ALLOC_EVENT))) {
        return true;
    }

    // bit0~9 devId
    int32_t devId = (flag & 0x3FF);
    int32_t moduleId = GetMallocModuleId(flag);
    MemOpSpace space = GetMemOpSpace(flag);
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord {};
    eventRecord.type = RecordType::MEMORY_RECORD;
    eventRecord.record.memoryRecord = CreateMemRecord(MemOpType::MALLOC, flag, space, addr, size);
    eventRecord.record.memoryRecord.devType = DeviceType::NPU;
    eventRecord.record.memoryRecord.devId = devId;
    eventRecord.record.memoryRecord.modid = moduleId;
    eventRecord.record.memoryRecord.recordIndex = ++recordIndex_;
    eventRecord.record.memoryRecord.kernelIndex = kernelLaunchRecordIndex_;
    GetOwner(eventRecord.record.memoryRecord.owner, sizeof(eventRecord.record.memoryRecord.owner));
    auto sendNums = ReportRecordEvent(eventRecord, head, stack);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportFree(uint64_t addr, CallStackString& stack)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    if (IsNeedSkip()) {
        return true;
    }

    BitField<decltype(config_.eventType)> eventType(config_.eventType);
    if (!eventType.checkBit(static_cast<size_t>(EventType::FREE_EVENT))) {
        return true;
    }

    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord {};
    eventRecord.type = RecordType::MEMORY_RECORD;
    eventRecord.record.memoryRecord = CreateMemRecord(MemOpType::FREE, FLAG_INVALID, MemOpSpace::INVALID, addr, 0);
    eventRecord.record.memoryRecord.devType = DeviceType::NPU;
    eventRecord.record.memoryRecord.devId = GD_INVALID_NUM;
    eventRecord.record.memoryRecord.modid = INVALID_MODID;
    eventRecord.record.memoryRecord.recordIndex = ++recordIndex_;
    eventRecord.record.memoryRecord.kernelIndex = kernelLaunchRecordIndex_;
    eventRecord.record.memoryRecord.owner[0] = '\0';
    auto sendNums = ReportRecordEvent(eventRecord, head, stack);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportHostMalloc(uint64_t addr, uint64_t size)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    BitField<decltype(config_.eventType)> eventType(config_.eventType);
    if (!eventType.checkBit(static_cast<size_t>(EventType::ALLOC_EVENT))) {
        return true;
    }

    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord {};
    eventRecord.type = RecordType::MEMORY_RECORD;
    eventRecord.record.memoryRecord = CreateMemRecord(
        MemOpType::MALLOC, FLAG_INVALID, MemOpSpace::INVALID, addr, size);
    eventRecord.record.memoryRecord.devType = DeviceType::CPU;
    eventRecord.record.memoryRecord.devId = 0;
    eventRecord.record.memoryRecord.modid = INVALID_MODID;
    eventRecord.record.memoryRecord.recordIndex = ++recordIndex_;
    eventRecord.record.memoryRecord.kernelIndex = kernelLaunchRecordIndex_;
    GetOwner(eventRecord.record.memoryRecord.owner, sizeof(eventRecord.record.memoryRecord.owner));
    auto sendNums = ReportRecordEvent(eventRecord, head);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}
 
bool EventReport::ReportHostFree(uint64_t addr)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    BitField<decltype(config_.eventType)> eventType(config_.eventType);
    if (!eventType.checkBit(static_cast<size_t>(EventType::FREE_EVENT))) {
        return true;
    }

    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord {};
    eventRecord.type = RecordType::MEMORY_RECORD;
    eventRecord.record.memoryRecord = CreateMemRecord(MemOpType::FREE, FLAG_INVALID, MemOpSpace::INVALID, addr, 0);
    eventRecord.record.memoryRecord.devType = DeviceType::CPU;
    eventRecord.record.memoryRecord.devId = 0;
    eventRecord.record.memoryRecord.modid = INVALID_MODID;
    eventRecord.record.memoryRecord.recordIndex = ++recordIndex_;
    eventRecord.record.memoryRecord.kernelIndex = kernelLaunchRecordIndex_;
    eventRecord.record.memoryRecord.owner[0] = '\0';
    auto sendNums = ReportRecordEvent(eventRecord, head);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

void EventReport::SetStepInfo(const MstxRecord &mstxRecord)
{
    if (mstxRecord.markType == MarkType::MARK_A) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (mstxRecord.markType == MarkType::RANGE_START_A) {
        if (strcmp(mstxRecord.markMessage, "step start") != 0) {
            return;
        }
        stepInfo_.currentStepId++;
        stepInfo_.inStepRange = true;
        stepInfo_.stepMarkRangeIdList.emplace_back(mstxRecord.rangeId);
        return;
    }

    if (mstxRecord.markType == MarkType::RANGE_END) {
        auto ret = find(stepInfo_.stepMarkRangeIdList.begin(), stepInfo_.stepMarkRangeIdList.end(), mstxRecord.rangeId);
        if (ret == stepInfo_.stepMarkRangeIdList.end()) {
            return;
        }
        stepInfo_.inStepRange = false;
        stepInfo_.stepMarkRangeIdList.erase(ret);
        return;
    }

    return;
}

bool EventReport::ReportMark(MstxRecord& mstxRecord, CallStackString& stack)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::MSTX_MARK_RECORD;
    mstxRecord.devId = devId;
    eventRecord.record.mstxRecord = mstxRecord;
    eventRecord.record.mstxRecord.pid = Utility::GetPid();
    eventRecord.record.mstxRecord.tid = Utility::GetTid();
    eventRecord.record.mstxRecord.timeStamp = Utility::GetTimeMicroseconds();
    eventRecord.record.mstxRecord.kernelIndex = kernelLaunchRecordIndex_;
    eventRecord.record.mstxRecord.recordIndex = ++recordIndex_;

    SetStepInfo(mstxRecord);
    eventRecord.record.mstxRecord.stepId = stepInfo_.currentStepId;
    auto sendNums = ReportRecordEvent(eventRecord, head, stack);

    // 通过有无固化语句判断是否要采集host侧内存数据
    {
        std::lock_guard<std::mutex> lock(rangeIdTableMutex_);
        uint64_t pid = eventRecord.record.mstxRecord.pid;
        uint64_t tid = eventRecord.record.mstxRecord.tid;
        if (mstxRecord.markType == MarkType::RANGE_START_A &&
            strcmp(mstxRecord.markMessage, "report host memory info start") == 0) {
            mstxRangeIdTables_[pid][tid] = mstxRecord.rangeId;
            CLIENT_INFO_LOG("[mark] Start report host memory info...");
            g_isReportHostMem = true;
        } else if (mstxRecord.markType == MarkType::RANGE_END &&
            mstxRangeIdTables_.find(pid) != mstxRangeIdTables_.end() &&
            mstxRangeIdTables_[pid].find(tid) != mstxRangeIdTables_[pid].end() &&
            mstxRangeIdTables_[pid][tid] == mstxRecord.rangeId) {
            mstxRangeIdTables_[pid].erase(tid);
            CLIENT_INFO_LOG("[mark] Stop report host memory info.");
            g_isReportHostMem = false;
        }
    }

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportAtenLaunch(AtenOpLaunchRecord &atenOpLaunchRecord, CallStackString& stack)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    if (IsNeedSkip()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::ATEN_OP_LAUNCH_RECORD;
    atenOpLaunchRecord.devId = devId;
    atenOpLaunchRecord.pid = Utility::GetPid();
    atenOpLaunchRecord.tid = Utility::GetTid();
    atenOpLaunchRecord.timestamp = Utility::GetTimeMicroseconds();
    atenOpLaunchRecord.recordIndex = ++recordIndex_;
    eventRecord.record.atenOpLaunchRecord = atenOpLaunchRecord;

    auto sendNums = ReportRecordEvent(eventRecord, head, stack);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportAtenAccess(MemAccessRecord &memAccessRecord, CallStackString& stack)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    if (IsNeedSkip()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::MEM_ACCESS_RECORD;
    memAccessRecord.pid = Utility::GetPid();
    memAccessRecord.tid = Utility::GetTid();
    memAccessRecord.timestamp = Utility::GetTimeMicroseconds();
    memAccessRecord.devId = devId;
    memAccessRecord.devType = DeviceType::NPU;
    memAccessRecord.recordIndex = ++recordIndex_;
    eventRecord.record.memAccessRecord = memAccessRecord;

    auto sendNums = ReportRecordEvent(eventRecord, head, stack);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportKernelLaunch(const AclnnKernelMapInfo &kernelLaunchInfo)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    if (IsNeedSkip()) {
        return true;
    }

    BitField<decltype(config_.eventType)> eventType(config_.eventType);
    BitField<decltype(config_.levelType)> levelType(config_.levelType);
    if (!eventType.checkBit(static_cast<size_t>(EventType::LAUNCH_EVENT)) ||
        !levelType.checkBit(static_cast<size_t>(LevelType::LEVEL_KERNEL))) {
        return true;
    }

    int32_t devId = std::get<0>(kernelLaunchInfo.taskKey);
    if (devId < 0) {
        if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
            CLIENT_ERROR_LOG("RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
        }
    }

    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::KERNEL_LAUNCH_RECORD;
    eventRecord.record.kernelLaunchRecord.timeStamp = kernelLaunchInfo.timeStamp;
    eventRecord.record.kernelLaunchRecord.pid = Utility::GetPid();
    eventRecord.record.kernelLaunchRecord.tid = Utility::GetTid();
    eventRecord.record.kernelLaunchRecord.devId = devId;
    eventRecord.record.kernelLaunchRecord.streamId = std::get<1>(kernelLaunchInfo.taskKey);
    eventRecord.record.kernelLaunchRecord.taskId = std::get<2>(kernelLaunchInfo.taskKey);
    eventRecord.record.kernelLaunchRecord.kernelLaunchIndex = ++kernelLaunchRecordIndex_;
    eventRecord.record.kernelLaunchRecord.recordIndex = ++recordIndex_;
    auto ret = strncpy_s(eventRecord.record.kernelLaunchRecord.kernelName,
        KERNELNAME_MAX_SIZE, kernelLaunchInfo.kernelName.c_str(), KERNELNAME_MAX_SIZE - 1);
    if (ret != EOK) {
        CLIENT_WARN_LOG("strncpy_s FAILED");
    }
    CallStackString stack;

    PacketHead head = {PacketType::RECORD};
    auto sendNums = ReportRecordEvent(eventRecord, head, stack);
    if (sendNums < 0) {
        return false;
    }

    g_isInReportFunction = false;
    return true;
}

bool EventReport::ReportKernelExcute(const TaskKey &key, std::string &name, uint64_t time, KernelEventType type)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }
    
    if (IsNeedSkip()) {
        return true;
    }

    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::KERNEL_EXCUTE_RECORD;
    eventRecord.record.kernelExcuteRecord.type = type;
    eventRecord.record.kernelExcuteRecord.devId = std::get<0>(key);
    eventRecord.record.kernelExcuteRecord.streamId = std::get<1>(key);
    eventRecord.record.kernelExcuteRecord.taskId = std::get<2>(key);
    eventRecord.record.kernelExcuteRecord.timeStamp = time;
    eventRecord.record.kernelExcuteRecord.recordIndex = ++recordIndex_;
    auto ret = strncpy_s(eventRecord.record.kernelExcuteRecord.kernelName,
        KERNELNAME_MAX_SIZE, name.c_str(), KERNELNAME_MAX_SIZE - 1);
    if (ret != EOK) {
        CLIENT_WARN_LOG("strncpy_s FAILED");
    }
    CallStackString stack;
    auto sendNums = ReportRecordEvent(eventRecord, head, stack);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}
bool EventReport::ReportAclItf(AclOpType aclOpType)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }
    
    if (IsNeedSkip()) {
        return true;
    }

    if (aclOpType == AclOpType::FINALIZE) {
        KernelEventTrace::GetInstance().EndKernelEventTrace();
    }
    int32_t devId = GD_INVALID_NUM;
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::ACL_ITF_RECORD;
    eventRecord.record.aclItfRecord = CreateAclItfRecord(aclOpType);
    eventRecord.record.aclItfRecord.devId = devId;
    eventRecord.record.aclItfRecord.recordIndex = ++recordIndex_;
    eventRecord.record.aclItfRecord.aclItfRecordIndex = ++aclItfRecordIndex_;
    eventRecord.record.aclItfRecord.kernelIndex = kernelLaunchRecordIndex_;
    CallStackString stack;
    auto sendNums = ReportRecordEvent(eventRecord, head, stack);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportAtbOpExecute(AtbOpExecuteRecord& atbOpExecuteRecord)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    if (IsNeedSkip()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::ATB_OP_EXECUTE_RECORD;
    atbOpExecuteRecord.devId = devId;
    atbOpExecuteRecord.pid = Utility::GetPid();
    atbOpExecuteRecord.tid = Utility::GetTid();
    atbOpExecuteRecord.timestamp = Utility::GetTimeMicroseconds();
    atbOpExecuteRecord.recordIndex = ++recordIndex_;
    eventRecord.record.atbOpExecuteRecord = atbOpExecuteRecord;

    CallStackString stack;
    auto sendNums = ReportRecordEvent(eventRecord, head, stack);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportAtbKernel(AtbKernelRecord& atbKernelRecord)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    if (IsNeedSkip()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::ATB_KERNEL_RECORD;
    atbKernelRecord.devId = devId;
    atbKernelRecord.pid = Utility::GetPid();
    atbKernelRecord.tid = Utility::GetTid();
    atbKernelRecord.timestamp = Utility::GetTimeMicroseconds();
    atbKernelRecord.recordIndex = ++recordIndex_;
    eventRecord.record.atbKernelRecord = atbKernelRecord;

    CallStackString stack;
    auto sendNums = ReportRecordEvent(eventRecord, head, stack);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportAtbAccessMemory(std::vector<MemAccessRecord>& memAccessRecords)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    if (IsNeedSkip()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    uint64_t pid = Utility::GetPid();
    uint64_t tid = Utility::GetTid();
    uint64_t timestamp = Utility::GetTimeMicroseconds();

    for (auto& record : memAccessRecords) {
        PacketHead head = {PacketType::RECORD};
        auto eventRecord = EventRecord{};
        eventRecord.type = RecordType::MEM_ACCESS_RECORD;
        record.pid = pid;
        record.tid = tid;
        record.timestamp = timestamp;
        record.devId = devId;
        record.devType = DeviceType::NPU;
        record.recordIndex = ++recordIndex_;
        eventRecord.record.memAccessRecord = record;
        CallStackString stack;
        auto sendNums = ReportRecordEvent(eventRecord, head, stack);
        if (sendNums < 0) {
            return false;
        }
    }

    g_isInReportFunction = false;
    return true;
}

}