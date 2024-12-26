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

namespace Leaks {


constexpr uint64_t MEM_MODULE_ID_BIT = 56;
constexpr uint64_t MEM_VIRT_BIT = 10;
constexpr uint64_t MEM_SVM_VAL = 0x0;
constexpr uint64_t MEM_DEV_VAL = 0x1;
constexpr uint64_t MEM_HOST_VAL = 0x2;
constexpr uint64_t MEM_DVPP_VAL = 0x3;

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
            Utility::LogError("No matching memType for %d .", memType);
    }
    return space;
}
inline int32_t GetMallocModuleId(unsigned long long flag)
{
    // bit56~63: model id
    return (flag & 0xFF00000000000000) >> MEM_MODULE_ID_BIT;
}

constexpr unsigned long long FLAG_INVALID = UINT64_MAX;

constexpr int32_t GD_INVALID_NUM = 9999;

constexpr int32_t INVALID_MODID = -1;

RTS_API rtError_t GetDevice(int32_t *devId)
{
    char const *sym = "rtGetDevice";
    using RtGetDevice = decltype(&GetDevice);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtGetDevice>(sym);
    if (vallina == nullptr) {
        Utility::LogError("vallina func get FAILED");
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

EventReport& EventReport::Instance(CommType type)
{
    static EventReport instance(type);
    return instance;
}

EventReport::EventReport(CommType type)
{
    (void)LocalProcess::GetInstance(type); // 连接server
    // 接受server端发送的消息
    std::string msg;
    // 默认10次重试
    LocalProcess::GetInstance(type).Wait(msg);
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
        if (currentStep_ == stepList.stepIdList[loop]) {
            return false;
        }
    }

    return true;
}

bool EventReport::ReportTorchNpu(TorchNpuRecord &torchNpuRecord)
{
    if (IsNeedSkip()) {
        return true;
    }
    PacketHead head = {PacketType::RECORD};
    EventRecord eventrecord;
    eventrecord.type = RecordType::TORCH_NPU_RECORD;
    eventrecord.record.torchNpuRecord = torchNpuRecord;
    eventrecord.record.torchNpuRecord.timeStamp = Utility::GetTimeMicroseconds();
    eventrecord.record.torchNpuRecord.devId = static_cast<int32_t>(torchNpuRecord.memoryUsage.deviceIndex);
    std::lock_guard<std::mutex> guard(mutex_);
    eventrecord.record.torchNpuRecord.recordIndex = ++recordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventrecord));
    Utility::LogInfo("TorchNpu Record, index: %u", recordIndex_);
    return (sendNums >= 0);
}

bool EventReport::ReportMalloc(uint64_t addr, uint64_t size, unsigned long long flag)
{
    if (IsNeedSkip()) {
        return true;
    }
    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
        Utility::LogError("RT_ERROR_INVALID_VALUE, %d!!!!!!!!!!!!", devId);
    }
    int32_t moduleId = GetMallocModuleId(flag);
    MemOpSpace space = GetMemOpSpace(flag);
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord {};
    eventRecord.type = RecordType::MEMORY_RECORD;
    eventRecord.record.memoryRecord = CreateMemRecord(MemOpType::MALLOC, flag, space, addr, size);
    eventRecord.record.memoryRecord.devId = devId;
    eventRecord.record.memoryRecord.modid = moduleId;
    std::lock_guard<std::mutex> guard(mutex_);
    eventRecord.record.memoryRecord.recordIndex = ++recordIndex_;
    eventRecord.record.memoryRecord.kernelIndex = kernelLaunchRecordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client malloc record, index: %u, addr: 0x%lx, size: %u, space: %u, flag: %llu",
        recordIndex_, addr, size, space, flag);
    return (sendNums >= 0);
}

bool EventReport::ReportFree(uint64_t addr)
{
    if (IsNeedSkip()) {
        return true;
    }
    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
        Utility::LogError("RT_ERROR_INVALID_VALUE, %d!!!!!!!!!!!!", devId);
    }
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord {};
    eventRecord.type = RecordType::MEMORY_RECORD;
    eventRecord.record.memoryRecord = CreateMemRecord(MemOpType::FREE, FLAG_INVALID, MemOpSpace::INVALID, addr, 0);
    eventRecord.record.memoryRecord.devId = devId;
    eventRecord.record.memoryRecord.modid = INVALID_MODID;
    std::lock_guard<std::mutex> guard(mutex_);
    eventRecord.record.memoryRecord.recordIndex = ++recordIndex_;
    eventRecord.record.memoryRecord.kernelIndex = kernelLaunchRecordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client free record, index: %u, addr: 0x%lx", recordIndex_, addr);
    return (sendNums >= 0);
}

bool EventReport::ReportMark(MstxRecord& mstxRecord)
{
    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
        Utility::LogError("RT_ERROR_INVALID_VALUE, %d!!!!!!!!!!!!", devId);
    }
    if (mstxRecord.streamId == -1) { // range end打点无需输出streamId信息，通过rangeId与start匹配
        Utility::LogInfo("this mark point message is %s", mstxRecord.markMessage);
    } else {
        Utility::LogInfo("this mark point message is %s, streamId is %d", mstxRecord.markMessage, mstxRecord.streamId);
    }
    Utility::LogInfo("this mark point id is %llu", mstxRecord.rangeId);
    
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::MSTX_MARK_RECORD;
    mstxRecord.devId = devId;
    eventRecord.record.mstxRecord = mstxRecord;
    eventRecord.record.mstxRecord.pid = Utility::GetPid();
    eventRecord.record.mstxRecord.tid = Utility::GetTid();
    eventRecord.record.mstxRecord.timeStamp = Utility::GetTimeMicroseconds();
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    
    std::lock_guard<std::mutex> guard(mutex_);
    if (mstxRecord.markType == MarkType::RANGE_START_A) {
        currentStep_ = mstxRecord.rangeId;
    }
    return (sendNums >= 0);
}

bool EventReport::ReportKernelLaunch(KernelLaunchRecord& kernelLaunchRecord, const void *hdl)
{
    if (IsNeedSkip()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
        Utility::LogError("RT_ERROR_INVALID_VALUE, %d!!!!!!!!!!!!", devId);
    }

    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::KERNEL_LAUNCH_RECORD;
    // 解析kernelname信息，默认不解析，打开-p开关后才解析
    if (config_.parseKernelName && strncpy_s(kernelLaunchRecord.kernelName, sizeof(kernelLaunchRecord.kernelName),
        GetNameFromBinary(hdl).c_str(), sizeof(kernelLaunchRecord.kernelName) - 1) != EOK) {
        Utility::LogError("strncpy_s FAILED");
    }
    eventRecord.record.kernelLaunchRecord = CreateKernelLaunchRecord(kernelLaunchRecord);
    eventRecord.record.kernelLaunchRecord.devId = devId;
    std::lock_guard<std::mutex> guard(mutex_);
    eventRecord.record.kernelLaunchRecord.kernelLaunchIndex = ++kernelLaunchRecordIndex_;
    eventRecord.record.kernelLaunchRecord.recordIndex = ++recordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client kernelLaunch record, name: %s, index: %u, type: %u, time: %u, streamId: %d, blockDim: %u",
        eventRecord.record.kernelLaunchRecord.kernelName,
        kernelLaunchRecordIndex_,
        eventRecord.record.kernelLaunchRecord.type,
        eventRecord.record.kernelLaunchRecord.timeStamp,
        eventRecord.record.kernelLaunchRecord.streamId,
        eventRecord.record.kernelLaunchRecord.blockDim);
    return (sendNums >= 0);
}

bool EventReport::ReportAclItf(AclOpType aclOpType)
{
    if (IsNeedSkip()) {
        return true;
    }

    int32_t devId = GD_INVALID_NUM;
    if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
        Utility::LogError("RT_ERROR_INVALID_VALUE, %d!!!!!!!!!!!!", devId);
    }

    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::ACL_ITF_RECORD;
    eventRecord.record.aclItfRecord = CreateAclItfRecord(aclOpType);
    eventRecord.record.aclItfRecord.devId = devId;
    std::lock_guard<std::mutex> guard(mutex_);
    eventRecord.record.aclItfRecord.recordIndex = ++recordIndex_;
    eventRecord.record.aclItfRecord.aclItfRecord = ++aclItfRecordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client aclItf record, index: %u, type: %u, time: %u",
        aclItfRecordIndex_,
        aclOpType,
        eventRecord.record.aclItfRecord.timeStamp);
    return (sendNums >= 0);
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
        Utility::LogError("PipeCall: get pipe failed");
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        Utility::LogError("PipeCall: create subprocess failed");
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
    auto it = HandleMapping::GetInstance().handleBinKernelMap_.find(hdl);
    if (it == HandleMapping::GetInstance().handleBinKernelMap_.end()) {
        Utility::LogError("kernel handle NOT registered in map");
        return kernelName;
    }
    std::vector<char> binary = it->second.bin;
    std::string kernelPath = "./kernel.o." + std::to_string(getpid());
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
    kernelName = ParseNameFromOutput(output);
    remove(kernelPath.c_str());
    return kernelName;
}

}