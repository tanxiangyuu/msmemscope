// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef EVENT_REPORT_H
#define EVENT_REPORT_H

#include <memory>
#include <string>
#include <mutex>
#include "client_process.h"
#include "kernel_hooks/runtime_hooks.h"
#include "record_info.h"
#include "config_info.h"

#include <thread>
#include <atomic>

constexpr mode_t REGULAR_MODE_MASK = 0177;

namespace Leaks {
/*
 * EventReport类主要功能：
 * 1. 将劫持记录的信息传回到工具进程
*/
class EventReport {
public:
    static EventReport& Instance(CommType type);
    bool ReportMalloc(uint64_t addr, uint64_t size, unsigned long long flag);
    bool ReportFree(uint64_t addr);
    bool ReportHostMalloc(uint64_t addr, uint64_t size);
    bool ReportHostFree(uint64_t addr);
    bool ReportKernelLaunch(KernelLaunchRecord& kernelLaunchRecord, const void *hdl);
    bool ReportAclItf(AclOpType aclOpType);
    bool ReportMark(MstxRecord &mstxRecord);
    bool ReportTorchNpu(TorchNpuRecord &torchNpuRecord);
    ~EventReport();
private:
    explicit EventReport(CommType type);
    std::atomic<uint64_t> recordIndex_;
    std::atomic<uint64_t> kernelLaunchRecordIndex_;
    std::atomic<uint64_t> aclItfRecordIndex_;
    bool IsNeedSkip();                      // 支持采集指定step
    bool IsReportHostMem();                 // 支持采集指定范围内的malloc和free信息
    uint64_t currentStep_ = 0;
    AnalysisConfig config_;
    std::vector<std::thread> parseThreads_;
    uint32_t maxThreadNum = 200;            // 最大同时运行线程数
    std::atomic<uint32_t> runningThreads;   // 同时运行线程数
    std::unordered_map<int32_t, uint64_t> mstxRangeIdTables_{};
    bool isReportHostMem_ = false;
    bool isInReportFunction_ = false;
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