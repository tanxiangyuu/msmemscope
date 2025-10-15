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
bool g_isReportHostMem = false;

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

bool GetDevice(int32_t *devId)
{
    char const *sym = "aclrtGetDeviceImpl";
    using AclrtGetDevice = aclError (*)(int32_t*);
    static AclrtGetDevice vallina = nullptr;
    if (vallina == nullptr) {
        vallina = VallinaSymbol<ACLImplLibLoader>::Instance().Get<AclrtGetDevice>(sym);
    }
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__) + ", try to get it in legacy way.");
        
        // 添加老版本的GetDevice逻辑，用于兼容情况如开放态场景
        char const *l_sym = "rtGetDevice";
        using RtGetDevice = rtError_t (*)(int32_t*);
        static RtGetDevice l_vallina = nullptr;
        if (l_vallina == nullptr) {
            l_vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtGetDevice>(l_sym);
        }
        if (l_vallina == nullptr) {
            CLIENT_ERROR_LOG("vallina func get FAILED in legacy way: " + std::string(__func__));
            return false;
        }

        rtError_t ret = l_vallina(devId);
        if (ret == RT_ERROR_INVALID_VALUE) {
            return false;
        }
        return true;
    }

    aclError ret = vallina(devId);
    if (ret != ACL_SUCCESS) {
        return false;
    }
    return true;
}

int EventReport::ReportRecordEvent(const RecordBuffer& record)
{
    std::string head = Serialize<PacketHead>(PacketHead{PacketType::RECORD, record.Size()});
    std::string buffer =  head + record.Get();
    auto sendNums = ClientProcess::GetInstance(LeaksCommType::SHARED_MEMORY).Notify(buffer);
    if (sendNums < 0) {
        isReceiveServerInfo_.store(false);
        std::cerr << "Process[" << Utility::GetPid() << "]: Client connection interrupted." << std::endl;
    }

    return sendNums;
}

EventReport& EventReport::Instance(LeaksCommType type)
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

Config EventReport::GetInitConfig()
{
    return initConfig_;
}

EventReport::EventReport(LeaksCommType type)
{
    Init();

    (void)ClientProcess::GetInstance(type); // 连接server
    std::string msg;
    uint32_t reTryTimes = 5; // 当前系统设置（setsockopt）的read超时时长是1s，默认至多尝试5次
    isReceiveServerInfo_ = (ClientProcess::GetInstance(type).Wait(msg, reTryTimes) > 0) ? true : false;
    Deserialize(msg, initConfig_);

    ClientProcess::GetInstance(type).SetLogLevel(static_cast<LogLv>(initConfig_.logLevel));

    RegisterRtProfileCallback();

    return;
}

EventReport::~EventReport()
{
    destroyed_.store(true);
}

bool EventReport::IsNeedSkip(int32_t devid)
{
    if (!GetConfig().collectAllNpu) {
        BitField<decltype(GetConfig().npuSlots)> npuSlots(GetConfig().npuSlots);
        if (devid != GD_INVALID_NUM && !npuSlots.checkBit(static_cast<size_t>(devid))) {
            return true;
        }
    }
    auto stepList = GetConfig().stepList;
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

bool EventReport::ReportAddrInfo(RecordBuffer &infoBuffer)
{
    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    GetDevice(&devId);
    if (IsNeedSkip(devId)) {
        return true;
    }

    AddrInfo* info = infoBuffer.Cast<AddrInfo>();
    info->type = RecordType::ADDR_INFO_RECORD;
    auto sendNums = ReportRecordEvent(infoBuffer);
    return (sendNums >= 0);
}

bool EventReport::ReportMemPoolRecord(RecordBuffer &memPoolRecordBuffer)
{
    if (!EventTraceManager::Instance().IsNeedTrace(RecordType::MEMORY_POOL_RECORD)) {
        return true;
    }
    
    if (!IsConnectToServer()) {
        return true;
    }

    MemPoolRecord* record = memPoolRecordBuffer.Cast<MemPoolRecord>();
    int32_t devId = static_cast<int32_t>(record->memoryUsage.deviceIndex);
    if (IsNeedSkip(devId)) {
        return true;
    }

    record->kernelIndex = kernelLaunchRecordIndex_;
    record->devId = devId;
    record->recordIndex = ++recordIndex_;
    auto sendNums = ReportRecordEvent(memPoolRecordBuffer);

    return (sendNums >= 0);
}

bool EventReport::ReportMalloc(uint64_t addr, uint64_t size, unsigned long long flag, CallStackString& stack)
{
    if (!IsConnectToServer()) {
        return true;
    }

    // bit0~9 devId
    int32_t devId = (flag & 0x3FF);
    if (IsNeedSkip(devId)) {
        return true;
    }

    MemOpSpace space = GetMemOpSpace(flag);
    // 不采集hal接口在host申请的pin memory
    if (space == MemOpSpace::HOST) {
        return true;
    }
    int32_t moduleId = GetMallocModuleId(flag);
    std::string owner = DescribeTrace::GetInstance().GetDescribe();
    TLVBlockType cStack = stack.cStack.empty() ? TLVBlockType::SKIP : TLVBlockType::CALL_STACK_C;
    TLVBlockType pyStack = stack.pyStack.empty() ? TLVBlockType::SKIP : TLVBlockType::CALL_STACK_PYTHON;
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemOpRecord>(
        TLVBlockType::MEM_OWNER, owner, cStack, stack.cStack, pyStack, stack.pyStack);
    MemOpRecord* record = buffer.Cast<MemOpRecord>();
    record->type = RecordType::MEMORY_RECORD;
    record->subtype = RecordSubType::MALLOC;
    record->flag = flag;
    record->space = space;
    record->addr = addr;
    record->memSize = size;
    record->devId = devId;
    record->modid = moduleId;
    record->recordIndex = ++recordIndex_;
    record->kernelIndex = kernelLaunchRecordIndex_;

    {
        if (!destroyed_.load()) {
            std::lock_guard<std::mutex> lock(mutex_);
            halPtrs_.insert(addr);
        }
    }

    auto sendNums = ReportRecordEvent(buffer);

    return (sendNums >= 0);
}

bool EventReport::ReportFree(uint64_t addr, CallStackString& stack)
{
    if (!IsConnectToServer()) {
        return true;
    }

    if (IsNeedSkip(GD_INVALID_NUM)) {
        return true;
    }

    {
        // 单例类析构之后不再访问其成员变量
        if (destroyed_.load()) {
            return true;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = halPtrs_.find(addr);
        if (it == halPtrs_.end()) {
            return true;
        }
        halPtrs_.erase(it);
    }

    TLVBlockType cStack = stack.cStack.empty() ? TLVBlockType::SKIP : TLVBlockType::CALL_STACK_C;
    TLVBlockType pyStack = stack.pyStack.empty() ? TLVBlockType::SKIP : TLVBlockType::CALL_STACK_PYTHON;
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemOpRecord>(cStack, stack.cStack, pyStack, stack.pyStack);
    MemOpRecord* record = buffer.Cast<MemOpRecord>();
    record->type = RecordType::MEMORY_RECORD;
    record->subtype = RecordSubType::FREE;
    record->flag = FLAG_INVALID;
    record->space = MemOpSpace::INVALID;
    record->addr = addr;
    record->memSize = 0;
    record->devId = GD_INVALID_NUM;
    record->modid = INVALID_MODID;
    record->recordIndex = ++recordIndex_;
    record->kernelIndex = kernelLaunchRecordIndex_;

    auto sendNums = ReportRecordEvent(buffer);

    return (sendNums >= 0);
}

void EventReport::SetStepInfo(const MstxRecord &mstxRecord)
{
    if (mstxRecord.markType == MarkType::MARK_A) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const TLVBlock* tlv = GetTlvBlock(mstxRecord, TLVBlockType::MARK_MESSAGE);
    std::string markMessage = tlv == nullptr ? "N/A" : tlv->data;
    if (mstxRecord.markType == MarkType::RANGE_START_A) {
        if (strcmp(markMessage.c_str(), "step start") != 0) {
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

bool EventReport::ReportMark(RecordBuffer &mstxRecordBuffer)
{
    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    MstxRecord* record = mstxRecordBuffer.Cast<MstxRecord>();
    record->type = RecordType::MSTX_MARK_RECORD;
    record->devId = devId;
    record->recordIndex = ++recordIndex_;
    record->kernelIndex = kernelLaunchRecordIndex_;

    SetStepInfo(*record);
    record->stepId = stepInfo_.currentStepId;

    if (IsNeedSkip(devId)) {
        return true;
    }

    auto sendNums = ReportRecordEvent(mstxRecordBuffer);

    return (sendNums >= 0);
}

bool EventReport::ReportAtenLaunch(RecordBuffer &atenOpLaunchRecordBuffer)
{
    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    if (IsNeedSkip(devId)) {
        return true;
    }

    AtenOpLaunchRecord* record = atenOpLaunchRecordBuffer.Cast<AtenOpLaunchRecord>();
    record->type = RecordType::ATEN_OP_LAUNCH_RECORD;
    record->devId = devId;
    record->recordIndex = ++recordIndex_;

    auto sendNums = ReportRecordEvent(atenOpLaunchRecordBuffer);

    return (sendNums >= 0);
}

bool EventReport::ReportAtenAccess(RecordBuffer &memAccessRecordBuffer)
{
    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    if (IsNeedSkip(devId)) {
        return true;
    }

    MemAccessRecord* record = memAccessRecordBuffer.Cast<MemAccessRecord>();
    record->type = RecordType::MEM_ACCESS_RECORD;
    record->devId = devId;
    record->recordIndex = ++recordIndex_;

    auto sendNums = ReportRecordEvent(memAccessRecordBuffer);

    return (sendNums >= 0);
}

bool EventReport::ReportKernelLaunch(const AclnnKernelMapInfo &kernelLaunchInfo)
{
    if (!EventTraceManager::Instance().IsNeedTrace(RecordType::KERNEL_LAUNCH_RECORD)) {
        return true;
    }

    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = std::get<0>(kernelLaunchInfo.taskKey);
    if (devId < 0) {
        if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
            CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
        }
    }

    if (IsNeedSkip(devId)) {
        return true;
    }

    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<KernelLaunchRecord>(
        TLVBlockType::KERNEL_NAME, kernelLaunchInfo.kernelName);
    KernelLaunchRecord* record = buffer.Cast<KernelLaunchRecord>();
    record->type = RecordType::KERNEL_LAUNCH_RECORD;
    record->devId = devId;
    record->streamId = std::get<1>(kernelLaunchInfo.taskKey);
    record->taskId = std::get<2>(kernelLaunchInfo.taskKey);
    record->kernelLaunchIndex = ++kernelLaunchRecordIndex_;
    record->recordIndex = ++recordIndex_;
    record->timestamp = kernelLaunchInfo.timestamp;
    auto sendNums = ReportRecordEvent(buffer);
    if (sendNums < 0) {
        return false;
    }

    return true;
}

bool EventReport::ReportKernelExcute(const TaskKey &key, std::string &name, uint64_t time, RecordSubType type)
{
    if (!EventTraceManager::Instance().IsNeedTrace(RecordType::KERNEL_EXCUTE_RECORD)) {
        return true;
    }

    if (!IsConnectToServer()) {
        return true;
    }
    
    if (IsNeedSkip(std::get<0>(key))) {
        return true;
    }

    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<KernelExcuteRecord>(TLVBlockType::KERNEL_NAME, name);
    KernelExcuteRecord* record = buffer.Cast<KernelExcuteRecord>();
    record->type = RecordType::KERNEL_EXCUTE_RECORD;
    record->subtype = type;
    record->devId = std::get<0>(key);
    record->streamId = std::get<1>(key);
    record->taskId = std::get<2>(key);
    record->recordIndex = ++recordIndex_;
    record->timestamp = time;
    auto sendNums = ReportRecordEvent(buffer);

    return (sendNums >= 0);
}
bool EventReport::ReportAclItf(RecordSubType subtype)
{
    if (!IsConnectToServer()) {
        return true;
    }
    
    if (IsNeedSkip(GD_INVALID_NUM)) {
        return true;
    }

    if (subtype == RecordSubType::FINALIZE) {
        KernelEventTrace::GetInstance().EndKernelEventTrace();
    }
    int32_t devId = GD_INVALID_NUM;

    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<AclItfRecord>();
    AclItfRecord* record = buffer.Cast<AclItfRecord>();
    record->type = RecordType::ACL_ITF_RECORD;
    record->subtype = subtype;
    record->devId = devId;
    record->recordIndex = ++recordIndex_;
    record->aclItfRecordIndex = ++aclItfRecordIndex_;
    record->kernelIndex = kernelLaunchRecordIndex_;
    auto sendNums = ReportRecordEvent(buffer);

    return (sendNums >= 0);
}

bool EventReport::ReportTraceStatus(const EventTraceStatus status)
{
    if (!IsConnectToServer()) {
        return true;
    }
    
    if (IsNeedSkip(GD_INVALID_NUM)) {
        return true;
    }

    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<TraceStatusRecord>();
    TraceStatusRecord* record = buffer.Cast<TraceStatusRecord>();
    record->type = RecordType::TRACE_STATUS_RECORD;
    record->devId = GD_INVALID_NUM;
    record->recordIndex = ++recordIndex_;
    record->status = static_cast<uint8_t>(status);
    auto sendNums = ReportRecordEvent(buffer);

    return (sendNums >= 0);
}

bool EventReport::ReportAtbOpExecute(char* name, uint32_t nameLength,
    char* attr, uint32_t attrLength, RecordSubType type)
{
    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    if (IsNeedSkip(devId)) {
        return true;
    }

    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<AtbOpExecuteRecord>(
        TLVBlockType::ATB_NAME, name, TLVBlockType::ATB_PARAMS, attr);
    AtbOpExecuteRecord* record = buffer.Cast<AtbOpExecuteRecord>();
    record->subtype = type;
    record->type = RecordType::ATB_OP_EXECUTE_RECORD;
    record->devId = devId;
    record->recordIndex = ++recordIndex_;

    auto sendNums = ReportRecordEvent(buffer);

    return (sendNums >= 0);
}

bool EventReport::ReportAtbKernel(char* name, uint32_t nameLength,
    char* attr, uint32_t attrLength, RecordSubType type)
{
    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    if (IsNeedSkip(devId)) {
        return true;
    }

    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<AtbKernelRecord>(
        TLVBlockType::ATB_NAME, name, TLVBlockType::ATB_PARAMS, attr);
    AtbKernelRecord* record = buffer.Cast<AtbKernelRecord>();
    record->subtype = type;
    record->type = RecordType::ATB_KERNEL_RECORD;
    record->devId = devId;
    record->recordIndex = ++recordIndex_;

    auto sendNums = ReportRecordEvent(buffer);

    return (sendNums >= 0);
}

bool EventReport::ReportAtbAccessMemory(char* name, char* attr, uint64_t addr, uint64_t size, AccessType type)
{
    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    if (IsNeedSkip(devId)) {
        return true;
    }

    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemAccessRecord>(
        TLVBlockType::OP_NAME, name, TLVBlockType::MEM_ATTR, attr);
    MemAccessRecord* record = buffer.Cast<MemAccessRecord>();
    record->addr = addr;
    record->memSize = size;
    record->eventType = type;

    record->memType = AccessMemType::ATB;
    record->type = RecordType::MEM_ACCESS_RECORD;
    record->devId = devId;
    record->recordIndex = ++recordIndex_;

    auto sendNums = ReportRecordEvent(buffer);

    return (sendNums >= 0);
}

}