// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef EVENT_REPORT_H
#define EVENT_REPORT_H

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include "client_process.h"
#include "kernel_hooks/runtime_hooks.h"
#include "record_info.h"
#include "config_info.h"
#include "protocol.h"

namespace Leaks {
extern thread_local bool g_isReportHostMem;
extern thread_local bool g_isInReportFunction;

constexpr mode_t REGULAR_MODE_MASK = 0177;
constexpr char ATEN_BEGIN_MSG[] = "leaks-aten-b:";
constexpr char ATEN_END_MSG[] = "leaks-aten-e:";
constexpr char ACCESS_MSG[] = "leaks-ac:";

struct MstxStepInfo {
    uint64_t currentStepId = 0;
    bool inStepRange = false; // 不在mstx_range_start和mstx_range_end之间的数据，不采集

    // 暂存用mstx标记Step的RangeId,与用mstx标记host采集的RangeId区分开
    // 可能存在多线程操作，例如线程A mstx_range_start,线程B mstx_range_start,线程A mstx_range_end,线程B mstx_range_end
    // 因此是数组
    std::vector<uint64_t> stepMarkRangeIdList;
};

/*
 * EventReport类主要功能：
 * 1. 将劫持记录的信息传回到工具进程
*/
class EventReport {
public:
    static EventReport& Instance(CommType type);
    bool ReportMalloc(uint64_t addr, uint64_t size, unsigned long long flag, CallStackString& stack);
    bool ReportFree(uint64_t addr, CallStackString& stack);
    bool ReportHostMalloc(uint64_t addr, uint64_t size);
    bool ReportHostFree(uint64_t addr);
    bool ReportKernelLaunch(KernelLaunchRecord& kernelLaunchRecord, const void *hdl);
    bool ReportAclItf(AclOpType aclOpType);
    bool ReportMark(MstxRecord &mstxRecord, CallStackString& stack);
    bool ReportTorchNpu(TorchNpuRecord &torchNpuRecord, CallStackString& stack);
    int ReportRecordEvent(EventRecord &record, PacketHead &head, CallStackString& stack);
    int ReportRecordEvent(EventRecord &record, PacketHead &head);
    Config GetConfig();
    bool ReportATBMemPoolRecord(AtbMemPoolRecord &record, CallStackString& stack);
    bool ReportAtbOpExecute(AtbOpExecuteRecord& atbOpExecuteRecord);
    bool ReportAtbKernel(AtbKernelRecord& atbKernelRecord);
    bool ReportAtbAccessMemory(std::vector<MemAccessRecord>& memAccessRecords);
    bool ReportAtenLaunch(AtenOpLaunchRecord &atenOpLaunchRecord, CallStackString& stack);
    bool ReportAtenAccess(MemAccessRecord &memAccessRecord, CallStackString& stack);
private:
    void Init();
    explicit EventReport(CommType type);
    ~EventReport();

    bool IsNeedSkip(); // 支持采集指定step
    void SetStepInfo(const MstxRecord &mstxRecord);

    // socket通信在某些场景下，client调用connect返回true并不一定代表真实连接成功
    // 这里以接收到server发过来消息为准
    bool IsConnectToServer();
private:
    std::atomic<uint64_t> recordIndex_;
    std::atomic<uint64_t> kernelLaunchRecordIndex_;
    std::atomic<uint64_t> aclItfRecordIndex_;

    MstxStepInfo stepInfo_;
    std::mutex mutex_;
    std::mutex threadMutex_;
    std::mutex rangeIdTableMutex_;

    Config config_;
    std::vector<std::thread> parseThreads_;
    std::atomic<uint32_t> runningThreads_;  // 同时运行线程数
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> mstxRangeIdTables_{};

    std::atomic<bool> isReceiveServerInfo_;
    std::map<const void*, std::string> hdlKernelNameMap_;
};

MemOpSpace GetMemOpSpace(unsigned long long flag);

inline int32_t GetMallocModuleId(unsigned long long flag);

extern "C" {
#ifndef RTS_API
#define RTS_API
#endif
RTS_API rtError_t GetDeviceID(int32_t *devId);
}

inline bool WriteBinary(std::string const &filename, char const *data, uint64_t length)
{
    if (!data) {
        return false;
    }
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    ofs.write(data, length);
    return ofs.good();
}

std::vector<char *> ToRawCArgv(std::vector<std::string> const &argv);
bool PipeCall(std::vector<std::string> const &cmd, std::string &output);
std::string ParseLine(std::string const &line);
std::string ParseNameFromOutput(std::string output);
std::string GetNameFromBinary(const void *hdl);

} // namespace Leaks
#endif