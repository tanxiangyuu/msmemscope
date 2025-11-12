// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef EVENT_REPORT_H
#define EVENT_REPORT_H

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <unordered_set>
#include "kernel_hooks/runtime_hooks.h"
#include "record_info.h"
#include "config_info.h"
#include "kernel_hooks/acl_hooks.h"
#include "kernel_hooks/kernel_event_trace.h"
#include "trace_manager/event_trace_manager.h"

#include "process.h"
#include "dump.h"

namespace Leaks {

constexpr mode_t REGULAR_MODE_MASK = 0177;
constexpr char ATEN_MSG[] = "leaks-aten-";
constexpr char ATEN_BEGIN_MSG[] = "b:";
constexpr char ATEN_END_MSG[] = "e:";
constexpr char ACCESS_MSG[] = "ac:";

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
    static EventReport& Instance(LeaksCommType type);
    bool ReportMalloc(uint64_t addr, uint64_t size, unsigned long long flag, CallStackString& stack);
    bool ReportFree(uint64_t addr, CallStackString& stack);
    bool ReportHostMalloc(uint64_t addr, uint64_t size, CallStackString& stack);
    bool ReportHostFree(uint64_t addr);
    bool ReportKernelLaunch(const AclnnKernelMapInfo &kernelLaunchInfo);
    bool ReportKernelExcute(const TaskKey &key, std::string &name, uint64_t time, RecordSubType type);
    bool ReportAclItf(RecordSubType subtype);
    bool ReportTraceStatus(const EventTraceStatus status);
    bool ReportMark(RecordBuffer &mstxRecordBuffer);
    void ReportRecordEvent(const RecordBuffer& record);
    bool ReportMemPoolRecord(RecordBuffer &memPoolRecordBuffer);
    bool ReportAtbOpExecute(char* name, uint32_t nameLength, char* attr, uint32_t attrLength, RecordSubType type);
    bool ReportAtbKernel(char* name, uint32_t nameLength, char* attr, uint32_t attrLength, RecordSubType type);
    bool ReportAtbAccessMemory(char* name, char* attr, uint64_t addr, uint64_t size, AccessType type);
    bool ReportAtenLaunch(RecordBuffer& atenOpLaunchRecordBuffer);
    bool ReportAtenAccess(RecordBuffer &memAccessRecordBuffer);
    bool ReportAddrInfo(RecordBuffer &infoBuffer);
    void UpdateAnalysisType();
private:
    void Init();
    explicit EventReport(LeaksCommType type);
    ~EventReport();

    bool IsNeedSkip(int32_t devid);
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

    Config initConfig_;

    std::unordered_set<uint64_t> halPtrs_;
    std::atomic<bool> destroyed_{false};
};

MemOpSpace GetMemOpSpace(unsigned long long flag);
bool GetDevice(int32_t *devId);

} // namespace Leaks
#endif