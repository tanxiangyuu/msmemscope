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
#include "handle_mapping.h"
#include "umask_guard.h"
#include "securec.h"
#include "bit_field.h"

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

KernelLaunchRecord CreateKernelLaunchRecord(KernelLaunchRecord kernelLaunchRecord)
{
    auto record = KernelLaunchRecord {};
    record = kernelLaunchRecord;
    record.timeStamp = Utility::GetTimeMicroseconds();
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
    runningThreads_.store(0);
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

EventReport::~EventReport()
{
    for (std::thread &t : parseThreads_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

bool EventReport::ReportTorchNpu(TorchNpuRecord &torchNpuRecord, CallStackString& stack)
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
    if (torchNpuRecord.memoryUsage.dataType == 0) {
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
    eventRecord.type = RecordType::TORCH_NPU_RECORD;
    eventRecord.record.torchNpuRecord = torchNpuRecord;
    eventRecord.record.torchNpuRecord.timeStamp = Utility::GetTimeMicroseconds();
    eventRecord.record.torchNpuRecord.kernelIndex = kernelLaunchRecordIndex_;
    eventRecord.record.torchNpuRecord.devId = static_cast<int32_t>(torchNpuRecord.memoryUsage.deviceIndex);
    eventRecord.record.torchNpuRecord.recordIndex = ++recordIndex_;
    auto sendNums = ReportRecordEvent(eventRecord, head, stack);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportATBMemPoolRecord(AtbMemPoolRecord &record, CallStackString& stack)
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
    if (record.memoryUsage.dataType == 0) {
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
    eventRecord.type = RecordType::ATB_MEMORY_POOL_RECORD;
    eventRecord.record.atbMemPoolRecord = record;
    eventRecord.record.atbMemPoolRecord.timeStamp = Utility::GetTimeMicroseconds();
    eventRecord.record.atbMemPoolRecord.kernelIndex = kernelLaunchRecordIndex_;
    eventRecord.record.atbMemPoolRecord.devId = static_cast<int32_t>(record.memoryUsage.deviceIndex);
    eventRecord.record.atbMemPoolRecord.recordIndex = ++recordIndex_;
    auto sendNums = ReportRecordEvent(eventRecord, head, stack);
    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportMalloc(
    uint64_t addr, uint64_t size, unsigned long long flag, CallStackString& stack)
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

bool EventReport::ReportKernelLaunch(KernelLaunchRecord& kernelLaunchRecord, const void *hdl)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    if (IsNeedSkip()) {
        return true;
    }

    BitField<decltype(config_.eventType)> eventType(config_.eventType);
    if (!eventType.checkBit(static_cast<size_t>(EventType::LAUNCH_EVENT))) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[kernellaunch] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::KERNEL_LAUNCH_RECORD;
    eventRecord.record.kernelLaunchRecord = CreateKernelLaunchRecord(kernelLaunchRecord);
    eventRecord.record.kernelLaunchRecord.devId = devId;
    eventRecord.record.kernelLaunchRecord.kernelLaunchIndex = ++kernelLaunchRecordIndex_;
    eventRecord.record.kernelLaunchRecord.recordIndex = ++recordIndex_;
    CallStackString stack;
    BitField<decltype(config_.levelType)> levelType(config_.levelType);
    if (levelType.checkBit(static_cast<size_t>(LevelType::LEVEL_KERNEL))) {
        std::string kernelName;
        {
            std::lock_guard<std::mutex> lock(threadMutex_);
            auto it = hdlKernelNameMap_.find(hdl);
            if (it != hdlKernelNameMap_.end()) {
                kernelName = it->second;
            }
        }
        if (!kernelName.empty()) {
            auto ret = strncpy_s(eventRecord.record.kernelLaunchRecord.kernelName,
                KERNELNAME_MAX_SIZE, kernelName.c_str(), KERNELNAME_MAX_SIZE - 1);
            if (ret != EOK) {
                CLIENT_WARN_LOG("strncpy_s FAILED");
            }
            PacketHead head = {PacketType::RECORD};
            auto sendNums = ReportRecordEvent(eventRecord, head, stack);
            if (sendNums < 0) {
                CLIENT_ERROR_LOG("rtKernelLaunch report FAILED");
                return false;
            }
            return true;
        }
        std::thread th = std::thread([eventRecord, hdl, this]()mutable {
            while (runningThreads_ >= MAX_THREAD_NUM) { // 达到最大线程数，等待直到有可用的线程
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 等待100ms
            }
            ++runningThreads_;
            std::string tempName = GetNameFromBinary(hdl);
            auto ret = strncpy_s(eventRecord.record.kernelLaunchRecord.kernelName,
                KERNELNAME_MAX_SIZE, tempName.c_str(), KERNELNAME_MAX_SIZE - 1);
            if (ret != EOK) {
                CLIENT_WARN_LOG("strncpy_s FAILED");
            }
            {
                std::lock_guard<std::mutex> lock(threadMutex_);
                hdlKernelNameMap_.insert({hdl, tempName});
            }
            PacketHead head = {PacketType::RECORD};
            CallStackString stack;
            auto sendNums = ReportRecordEvent(eventRecord, head, stack);
            if (sendNums < 0) {
                CLIENT_ERROR_LOG("rtKernelLaunch report FAILED");
                return;
            }
            --runningThreads_;
        });
        parseThreads_.emplace_back(std::move(th));
    } else {
        PacketHead head = {PacketType::RECORD};
        auto sendNums = ReportRecordEvent(eventRecord, head, stack);
        if (sendNums < 0) {
            return false;
        }
    }

    g_isInReportFunction = false;
    return true;
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

std::vector<char *> ToRawCArgv(std::vector<std::string> const &argv)
{
    std::vector<char *> rawArgv;
    for (auto const &arg: argv) {
        rawArgv.emplace_back(const_cast<char *>(arg.data()));
    }
    rawArgv.emplace_back(nullptr);
    return rawArgv;
}

bool PipeCall(std::vector<std::string> const &cmd, std::string &output)
{
    int pipeStdout[2];
    if (pipe(pipeStdout) != 0) {
        CLIENT_ERROR_LOG("PipeCall: get pipe failed");
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        CLIENT_ERROR_LOG("PipeCall: create subprocess failed");
        return false;
    } else if (pid == 0) {
        dup2(pipeStdout[1], STDOUT_FILENO);
        close(pipeStdout[0]);
        close(pipeStdout[1]);
        // llvm-objdump解析kernel.o文件中的二进制数据
        execvp(cmd[0].c_str(), ToRawCArgv(cmd).data());
        _exit(EXIT_FAILURE);
    } else {
        close(pipeStdout[1]);
        static constexpr std::size_t bufLen = 256UL;
        char buf[bufLen] = {'\0'};
        ssize_t nBytes = 0L;
        for (; (nBytes = read(pipeStdout[0], buf, bufLen)) > 0L;) {
            output.append(buf, static_cast<std::size_t>(nBytes));
        }
        close(pipeStdout[0]);
        int status;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
    return true;
}

std::string ParseLine(std::string const &line)
{
    std::vector<std::string> items;
    Utility::Split(line, std::back_inserter(items), " ");
    if (items.size() < 5UL) {
        return "";
    }
    constexpr std::size_t scopeIdx = 1UL;
    constexpr std::size_t symbolKindIdx = 2UL;
    if (items[scopeIdx] != "g" || items[symbolKindIdx] != "F") {
        return "";
    }
    constexpr std::size_t kernelNameIdx = 4UL;
    std::string kernelName = items[kernelNameIdx];
    if (Utility::EndWith(kernelName, "_mix_aic") ||
        Utility::EndWith(kernelName, "_mix_aiv")) {
        kernelName = kernelName.substr(0UL, kernelName.length() - 8UL);
    }

    items.clear();
    Utility::Split(kernelName, std::back_inserter(items), "_");
    if (items.size() < 2UL) {
        return "";
    }
    std::string kernelNamePrefix = items[0] + "_" + items[1];
    return kernelNamePrefix;
}

std::string ParseNameFromOutput(std::string output)
{
    std::string kernelName;
    std::vector<std::string> lines;
    Utility::Split(output, std::back_inserter(lines), "\n");

    // skip headers
    auto it = lines.cbegin();
    for (; it != lines.cend(); ++it) {
        if (it->find("SYMBOL TABLE:") != std::string::npos) {
            break;
        }
    }

    if (it == lines.cend()) {
        return kernelName;
    }
    ++it;

    for (; it != lines.cend(); ++it) {
        // 解析每一行符号表数据，取出kernelname
        kernelName = ParseLine(*it);
        if (!kernelName.empty()) {
            return kernelName;
        }
    }
    return kernelName;
}

std::string GetNameFromBinary(const void *hdl)
{
    std::string kernelName;
    std::vector<char> binary = HandleMapping::GetInstance().BinKernelMapFind(hdl);
    if (binary.empty()) {
        return kernelName;
    }
    auto time = Utility::GetTimeNanoseconds();
    std::string kernelPath = "./kernel.o." + std::to_string(time);
    {
        Utility::UmaskGuard umaskGuard(REGULAR_MODE_MASK);
        if (!WriteBinary(kernelPath, binary.data(), binary.size())) {
            return kernelName;
        }
    }
    std::vector<std::string> cmd = {
        "llvm-objdump",
        "-t",
        kernelPath
    };

    std::string output;
    bool ret = PipeCall(cmd, output);
    if (!ret) {
        CLIENT_ERROR_LOG("pipe call failed!");
        return kernelName;
    }
    kernelName = ParseNameFromOutput(output);
    remove(kernelPath.c_str());
    return kernelName;
}

}