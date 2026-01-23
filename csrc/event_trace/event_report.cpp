/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#include "event_report.h"
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include "log.h"
#include "utils.h"
#include "vallina_symbol.h"
#include "ustring.h"
#include "umask_guard.h"
#include "securec.h"
#include "bit_field.h"
#include "kernel_hooks/runtime_prof_api.h"
#include "describe_trace.h"
#include "decompose_analyzer.h"
#include "inefficient_analyzer.h"
#include "json_manager.h"

namespace MemScope {
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
            LOG_ERROR("No matching memType for " + std::to_string(memType));
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
        LOG_ERROR("vallina func get FAILED: " + std::string(__func__) + ", try to get it in legacy way.");
        
        // 添加老版本的GetDevice逻辑，用于兼容情况如开放态场景
        char const *l_sym = "rtGetDevice";
        using RtGetDevice = rtError_t (*)(int32_t*);
        static RtGetDevice l_vallina = nullptr;
        if (l_vallina == nullptr) {
            l_vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtGetDevice>(l_sym);
        }
        if (l_vallina == nullptr) {
            LOG_ERROR("vallina func get FAILED in legacy way: " + std::string(__func__));
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

bool GetDeviceMemInfo(size_t &free, size_t &total)
{
    using func = decltype(&aclrtGetMemInfo);
    static auto vallina = VallinaSymbol<AclLibLoader>::Instance().Get<func>("aclrtGetMemInfo");
    if (vallina == nullptr) {
        LOG_ERROR("Get aclrtGetMemInfo func ptr failed");
        return false;
    }

    int ret = vallina(ACL_HBM_MEM, &free, &total);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("Get device mem info failed, ret is " + std::to_string(ret));
        return false;
    }
    return true;
}

void EventReport::ReportRecordEvent(const RecordBuffer& record)
{
    Process::GetInstance(initConfig_).RecordHandler(record);
}

EventReport& EventReport::Instance(MemScopeCommType type)
{
    static EventReport instance(type);
    return instance;
}

void EventReport::Init()
{
    recordIndex_.store(0);
    kernelLaunchRecordIndex_.store(0);
    aclItfRecordIndex_.store(0);
    pyStepId_.store(0);
}

EventReport::EventReport(MemScopeCommType type)
{
    Init();
    initConfig_ = GetConfig();

    // subscribe订阅
    BitField<decltype(initConfig_.analysisType)> analysisType(initConfig_.analysisType);
    if (analysisType.checkBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        DecomposeAnalyzer::GetInstance();
    }
    if (analysisType.checkBit(static_cast<size_t>(AnalysisType::INEFFICIENCY_ANALYSIS))) {
        InefficientAnalyzer::GetInstance();
    }
    Dump::GetInstance(initConfig_);
    LOG_INFO("LOG INIT");
    RegisterRtProfileCallback();

    return;
}

void EventReport::UpdateAnalysisType()
{
    initConfig_ = GetConfig();
    BitField<decltype(initConfig_.analysisType)> analysisType(initConfig_.analysisType);

    // 根据config确认是否订阅或者取消订阅
    if (analysisType.checkBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        DecomposeAnalyzer::GetInstance().Subscribe();
    } else {
        DecomposeAnalyzer::GetInstance().UnSubscribe();
    }

    if (analysisType.checkBit(static_cast<size_t>(AnalysisType::INEFFICIENCY_ANALYSIS))) {
        InefficientAnalyzer::GetInstance().Subscribe();
    } else {
        InefficientAnalyzer::GetInstance().UnSubscribe();
    }
}

EventReport::~EventReport()
{
    destroyed_.store(true);
}

bool EventReport::IsNeedSkip(int32_t devid)
{
    // 是否为指定卡
    if (!GetConfig().collectAllNpu) {
        BitField<decltype(GetConfig().npuSlots)> npuSlots(GetConfig().npuSlots);
        if (devid != GD_INVALID_NUM && !npuSlots.checkBit(static_cast<size_t>(devid))) {
            return true;
        }
    }

    // 目前仅命令行支持选择--steps，因此当存在stepList时代表启用了命令行，我们不推荐同时使用命令行和python接口。这里不考虑
    // msmemscope.step()接口所带来的的step信息。
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

bool EventReport::ReportAddrInfo(RecordBuffer &infoBuffer)
{
    int32_t devId = GD_INVALID_NUM;
    GetDevice(&devId);
    if (IsNeedSkip(devId)) {
        return true;
    }

    AddrInfo* info = infoBuffer.Cast<AddrInfo>();
    info->type = RecordType::ADDR_INFO_RECORD;
    ReportRecordEvent(infoBuffer);
    return true;
}

bool EventReport::ReportPyStepRecord()
{
    int32_t devId = GD_INVALID_NUM;
    GetDevice(&devId);
    if (IsNeedSkip(devId)) {
        return true;
    }
 
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<PyStepRecord>();
    PyStepRecord* info = buffer.Cast<PyStepRecord>();
    info->type = RecordType::PY_STEP_RECORD;
    info->recordIndex = ++recordIndex_;
    info->stepId = ++pyStepId_;
    info->devId = devId;
    ReportRecordEvent(buffer);
    return true;
}
 
bool EventReport::ReportMemPoolRecord(RecordBuffer &memPoolRecordBuffer)
{
    if (!EventTraceManager::Instance().IsTracingEnabled() ||
        !EventTraceManager::Instance().ShouldTraceType(RecordType::MEMORY_POOL_RECORD)) {
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
    ReportRecordEvent(memPoolRecordBuffer);

    return true;
}

bool EventReport::ReportMalloc(uint64_t addr, uint64_t size, unsigned long long flag, CallStackString& stack)
{
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

    ReportRecordEvent(buffer);

    return true;
}

bool EventReport::ReportFree(uint64_t addr, CallStackString& stack)
{
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

    ReportRecordEvent(buffer);

    return true;
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
    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        LOG_ERROR("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
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

    ReportRecordEvent(mstxRecordBuffer);

    return true;
}

bool EventReport::ReportAtenLaunch(RecordBuffer &atenOpLaunchRecordBuffer)
{
    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        LOG_ERROR("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    if (IsNeedSkip(devId)) {
        return true;
    }

    AtenOpLaunchRecord* record = atenOpLaunchRecordBuffer.Cast<AtenOpLaunchRecord>();
    record->type = RecordType::ATEN_OP_LAUNCH_RECORD;
    record->devId = devId;
    record->recordIndex = ++recordIndex_;

    ReportRecordEvent(atenOpLaunchRecordBuffer);

    return true;
}

bool EventReport::ReportAtenAccess(RecordBuffer &memAccessRecordBuffer)
{
    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        LOG_ERROR("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    if (IsNeedSkip(devId)) {
        return true;
    }

    MemAccessRecord* record = memAccessRecordBuffer.Cast<MemAccessRecord>();
    record->type = RecordType::MEM_ACCESS_RECORD;
    record->devId = devId;
    record->recordIndex = ++recordIndex_;

    ReportRecordEvent(memAccessRecordBuffer);

    return true;
}

bool EventReport::ReportKernelLaunch(const AclnnKernelMapInfo &kernelLaunchInfo)
{
    if (!EventTraceManager::Instance().IsTracingEnabled() ||
        !EventTraceManager::Instance().ShouldTraceType(RecordType::KERNEL_LAUNCH_RECORD)) {
        return true;
    }

    int32_t devId = std::get<0>(kernelLaunchInfo.taskKey);
    if (devId < 0) {
        if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
            LOG_ERROR("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
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

    ReportRecordEvent(buffer);

    return true;
}

bool EventReport::ReportKernelExcute(const TaskKey &key, std::string &name, uint64_t time, RecordSubType type)
{
    if (!EventTraceManager::Instance().IsTracingEnabled() ||
        !EventTraceManager::Instance().ShouldTraceType(RecordType::KERNEL_EXCUTE_RECORD)) {
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

    ReportRecordEvent(buffer);

    return true;
}
bool EventReport::ReportAclItf(RecordSubType subtype)
{
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

    ReportRecordEvent(buffer);

    return true;
}

bool EventReport::ReportTraceStatus(const EventTraceStatus status)
{
    if (IsNeedSkip(GD_INVALID_NUM)) {
        return true;
    }

    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<TraceStatusRecord>();
    TraceStatusRecord* record = buffer.Cast<TraceStatusRecord>();
    record->type = RecordType::TRACE_STATUS_RECORD;
    record->devId = GD_INVALID_NUM;
    record->recordIndex = ++recordIndex_;
    record->status = static_cast<uint8_t>(status);

    ReportRecordEvent(buffer);

    return true;
}

bool EventReport::ReportAtbOpExecute(char* name, uint32_t nameLength,
    char* attr, uint32_t attrLength, RecordSubType type)
{
    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        LOG_ERROR("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
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

    ReportRecordEvent(buffer);

    return true;
}

bool EventReport::ReportAtbKernel(char* name, uint32_t nameLength,
    char* attr, uint32_t attrLength, RecordSubType type)
{
    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        LOG_ERROR("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
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

    ReportRecordEvent(buffer);

    return true;
}

bool EventReport::ReportAtbAccessMemory(char* name, char* attr, uint64_t addr, uint64_t size, AccessType type)
{
    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        LOG_ERROR("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
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

    ReportRecordEvent(buffer);

    return true;
}

bool EventReport::ReportMemorySnapshot(const MemorySnapshotRecord& memory_info)
{
    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        LOG_ERROR("[mark] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }

    if (IsNeedSkip(devId)) {
        return true;
    }

    // 创建内存快照记录
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemorySnapshotRecord>();
    MemorySnapshotRecord* record = buffer.Cast<MemorySnapshotRecord>();
    record->device = memory_info.device;
    record->memory_reserved = memory_info.memory_reserved;
    record->max_memory_reserved = memory_info.max_memory_reserved;
    record->memory_allocated = memory_info.memory_allocated;
    record->max_memory_allocated = memory_info.max_memory_allocated;
    record->total_memory = memory_info.total_memory;
    record->free_memory = memory_info.free_memory;

    std::string snapshot_name = memory_info.name;
    strncpy_s(record->name, sizeof(record->name), snapshot_name.c_str(), snapshot_name.length());

    record->type = RecordType::SNAPSHOT_EVENT;
    record->devId = devId;
    record->recordIndex = ++recordIndex_;

    ReportRecordEvent(buffer);

    return true;
}

}