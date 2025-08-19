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
thread_local bool g_isInReportFunction = false;
static std::unordered_set<uint64_t> g_halPtrs;

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

aclError GetDevice(int32_t *devId)
{
    char const *sym = "aclrtGetDeviceImpl";
    using AclrtGetDevice = decltype(&GetDevice);
    static AclrtGetDevice vallina = nullptr;
    if (vallina == nullptr) {
        vallina = VallinaSymbol<ACLImplLibLoader>::Instance().Get<AclrtGetDevice>(sym);
    }
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return ACL_ERROR_RT_FAILURE;
    }
    aclError ret = vallina(devId);
    return ret;
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
    g_isInReportFunction = true;
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
    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportMemPoolRecord(RecordBuffer &memPoolRecordBuffer)
{
    g_isInReportFunction = true;

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

    g_isInReportFunction = false;

    return (sendNums >= 0);
}

bool EventReport::ReportMalloc(uint64_t addr, uint64_t size, unsigned long long flag, CallStackString& stack)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    // bit0~9 devId
    int32_t devId = (flag & 0x3FF);
    if (IsNeedSkip(devId)) {
        return true;
    }

    MemOpSpace space = GetMemOpSpace(flag);
    if (space == MemOpSpace::HOST && !GetConfig().collectCpu) {
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
    record->devType = DeviceType::NPU;
    record->devId = devId;
    record->modid = moduleId;
    record->recordIndex = ++recordIndex_;
    record->kernelIndex = kernelLaunchRecordIndex_;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        g_halPtrs.insert(addr);
    }

    auto sendNums = ReportRecordEvent(buffer);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportFree(uint64_t addr, CallStackString& stack)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    if (IsNeedSkip(GD_INVALID_NUM)) {
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = g_halPtrs.find(addr);
        if (it == g_halPtrs.end()) {
            return true;
        }
        g_halPtrs.erase(it);
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
    record->devType = DeviceType::NPU;
    record->devId = GD_INVALID_NUM;
    record->modid = INVALID_MODID;
    record->recordIndex = ++recordIndex_;
    record->kernelIndex = kernelLaunchRecordIndex_;

    auto sendNums = ReportRecordEvent(buffer);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportHostMalloc(uint64_t addr, uint64_t size, CallStackString& stack)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    if (!GetConfig().collectCpu) {
        return true;
    }

    TLVBlockType cStack = stack.cStack.empty() ? TLVBlockType::SKIP : TLVBlockType::CALL_STACK_C;
    TLVBlockType pyStack = stack.pyStack.empty() ? TLVBlockType::SKIP : TLVBlockType::CALL_STACK_PYTHON;

    std::string owner = DescribeTrace::GetInstance().GetDescribe();
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemOpRecord>(TLVBlockType::MEM_OWNER, owner,
        cStack, stack.cStack, pyStack, stack.pyStack);
    MemOpRecord* record = buffer.Cast<MemOpRecord>();
    record->type = RecordType::MEMORY_RECORD;
    record->subtype = RecordSubType::MALLOC;
    record->flag = FLAG_INVALID;
    record->space = MemOpSpace::INVALID;
    record->addr = addr;
    record->memSize = size;
    record->devType = DeviceType::CPU;
    record->devId = 0;
    record->modid = INVALID_MODID;
    record->recordIndex = ++recordIndex_;
    record->kernelIndex = kernelLaunchRecordIndex_;

    auto sendNums = ReportRecordEvent(buffer);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}
 
bool EventReport::ReportHostFree(uint64_t addr)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    if (!GetConfig().collectCpu) {
        return true;
    }

    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemOpRecord>();
    MemOpRecord* record = buffer.Cast<MemOpRecord>();
    record->type = RecordType::MEMORY_RECORD;
    record->subtype = RecordSubType::FREE;
    record->flag = FLAG_INVALID;
    record->space = MemOpSpace::INVALID;
    record->addr = addr;
    record->memSize = 0;
    record->devType = DeviceType::CPU;
    record->devId = 0;
    record->modid = INVALID_MODID;
    record->recordIndex = ++recordIndex_;
    record->kernelIndex = kernelLaunchRecordIndex_;
    auto sendNums = ReportRecordEvent(buffer);

    g_isInReportFunction = false;
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
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) != ACL_SUCCESS || devId == GD_INVALID_NUM) {
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

    // 通过有无固化语句判断是否要采集host侧内存数据
    {
        std::lock_guard<std::mutex> lock(rangeIdTableMutex_);
        const TLVBlock* tlv = GetTlvBlock(*record, TLVBlockType::MARK_MESSAGE);
        std::string markMessage = tlv == nullptr ? "N/A" : tlv->data;
        uint64_t pid = record->pid;
        uint64_t tid = record->tid;
        if (record->markType == MarkType::RANGE_START_A &&
            strcmp(markMessage.c_str(), "report host memory info start") == 0) {
            mstxRangeIdTables_[pid][tid] = record->rangeId;
            CLIENT_INFO_LOG("[mark] Start report host memory info...");
            Config config = GetConfig();
            config.collectCpu = true;
            ConfigManager::Instance().SetConfig(config);
        } else if (record->markType == MarkType::RANGE_END &&
            mstxRangeIdTables_.find(pid) != mstxRangeIdTables_.end() &&
            mstxRangeIdTables_[pid].find(tid) != mstxRangeIdTables_[pid].end() &&
            mstxRangeIdTables_[pid][tid] == record->rangeId) {
            mstxRangeIdTables_[pid].erase(tid);
            CLIENT_INFO_LOG("[mark] Stop report host memory info.");
            Config config = GetConfig();
            config.collectCpu = false;
            ConfigManager::Instance().SetConfig(config);
        }
    }

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportAtenLaunch(RecordBuffer &atenOpLaunchRecordBuffer)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) != ACL_SUCCESS || devId == GD_INVALID_NUM) {
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

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportAtenAccess(RecordBuffer &memAccessRecordBuffer)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) != ACL_SUCCESS || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    if (IsNeedSkip(devId)) {
        return true;
    }

    MemAccessRecord* record = memAccessRecordBuffer.Cast<MemAccessRecord>();
    record->type = RecordType::MEM_ACCESS_RECORD;
    record->devId = devId;
    record->devType = DeviceType::NPU;
    record->recordIndex = ++recordIndex_;

    auto sendNums = ReportRecordEvent(memAccessRecordBuffer);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportKernelLaunch(const AclnnKernelMapInfo &kernelLaunchInfo)
{
    g_isInReportFunction = true;

    if (!EventTraceManager::Instance().IsNeedTrace(RecordType::KERNEL_LAUNCH_RECORD)) {
        return true;
    }

    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = std::get<0>(kernelLaunchInfo.taskKey);
    if (devId < 0) {
        if (GetDevice(&devId) != ACL_SUCCESS || devId == GD_INVALID_NUM) {
            CLIENT_ERROR_LOG("RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
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

    g_isInReportFunction = false;
    return true;
}

bool EventReport::ReportKernelExcute(const TaskKey &key, std::string &name, uint64_t time, RecordSubType type)
{
    g_isInReportFunction = true;

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

    g_isInReportFunction = false;
    return (sendNums >= 0);
}
bool EventReport::ReportAclItf(RecordSubType subtype)
{
    g_isInReportFunction = true;

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

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportTraceStatus(const EventTraceStatus status)
{
    g_isInReportFunction = true;

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

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportAtbOpExecute(RecordBuffer& atbOpExecuteRecordBuffer)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) != ACL_SUCCESS || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    if (IsNeedSkip(devId)) {
        return true;
    }

    AtbOpExecuteRecord* record = atbOpExecuteRecordBuffer.Cast<AtbOpExecuteRecord>();
    record->type = RecordType::ATB_OP_EXECUTE_RECORD;
    record->devId = devId;
    record->recordIndex = ++recordIndex_;

    auto sendNums = ReportRecordEvent(atbOpExecuteRecordBuffer);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportAtbKernel(RecordBuffer& atbKernelRecordBuffer)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) != ACL_SUCCESS || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    if (IsNeedSkip(devId)) {
        return true;
    }

    AtbKernelRecord* record = atbKernelRecordBuffer.Cast<AtbKernelRecord>();
    record->type = RecordType::ATB_KERNEL_RECORD;
    record->devId = devId;
    record->recordIndex = ++recordIndex_;

    auto sendNums = ReportRecordEvent(atbKernelRecordBuffer);

    g_isInReportFunction = false;
    return (sendNums >= 0);
}

bool EventReport::ReportAtbAccessMemory(std::vector<RecordBuffer>& memAccessRecordBuffers)
{
    g_isInReportFunction = true;

    if (!IsConnectToServer()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) != ACL_SUCCESS || devId == GD_INVALID_NUM) {
        CLIENT_ERROR_LOG("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    if (IsNeedSkip(devId)) {
        return true;
    }

    for (auto& buffer : memAccessRecordBuffers) {
        MemAccessRecord* record = buffer.Cast<MemAccessRecord>();
        record->type = RecordType::MEM_ACCESS_RECORD;
        record->devId = devId;
        record->devType = DeviceType::NPU;
        record->recordIndex = ++recordIndex_;

        auto sendNums = ReportRecordEvent(buffer);
        if (sendNums < 0) {
            return false;
        }
    }

    g_isInReportFunction = false;
    return true;
}

}