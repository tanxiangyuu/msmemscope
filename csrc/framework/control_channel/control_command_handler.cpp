/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#include "control_channel/control_command_handler.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "analysis/event_dispatcher.h"
#include "analysis/host_leak_analyzer.h"
#include "analysis/memory_state_manager.h"
#include "analysis/state_manager.h"
#include "control_channel/control_protocol.h"
#include "event_trace/event_report.h"
#include "trace_manager/event_trace_manager.h"
#include "utility/comm_def.h"
#include "utility/loaded_so.h"
#include "utility/log.h"
#include "utility/utils.h"

namespace MemScope
{

namespace
{
// 字节格式化(与host_leak_analyzer.cpp同款算法本地复制,控制字低频无下沉必要)
std::string FormatSizeAbbrev(double value)
{
    constexpr double kB = 1024.0;
    constexpr double kMB = 1024.0 * 1024.0;
    constexpr double kGB = 1024.0 * 1024.0 * 1024.0;
    char buf[64];
    if (value >= kGB)
    {
        std::snprintf(buf, sizeof(buf), "%.2f GB", value / kGB);
    }
    else if (value >= kMB)
    {
        std::snprintf(buf, sizeof(buf), "%.2f MB", value / kMB);
    }
    else if (value >= kB)
    {
        std::snprintf(buf, sizeof(buf), "%.2f KB", value / kB);
    }
    else
    {
        std::snprintf(buf, sizeof(buf), "%.0f B", value);
    }
    return std::string(buf);
}

// --pool 池名(小写)→PoolType;未知返回INVALID
PoolType PoolNameToType(const std::string& name)
{
    if (name == "host")
    {
        return PoolType::HOST;
    }
    if (name == "hal")
    {
        return PoolType::HAL;
    }
    if (name == "pta")
    {
        return PoolType::PTA_CACHING;
    }
    if (name == "pta_workspace")
    {
        return PoolType::PTA_WORKSPACE;
    }
    if (name == "atb")
    {
        return PoolType::ATB;
    }
    if (name == "mindspore")
    {
        return PoolType::MINDSPORE;
    }
    return PoolType::INVALID;
}

// analysis位图→名字串(逗号分隔);0返回"none"
std::string AnalysisTypeToNames(uint8_t bits)
{
    std::vector<std::string> names;
    if ((bits & (1u << static_cast<uint8_t>(AnalysisType::LEAKS_ANALYSIS))) != 0)
    {
        names.push_back("leaks");
    }
    if ((bits & (1u << static_cast<uint8_t>(AnalysisType::DECOMPOSE_ANALYSIS))) != 0)
    {
        names.push_back("decompose");
    }
    if ((bits & (1u << static_cast<uint8_t>(AnalysisType::INEFFICIENCY_ANALYSIS))) != 0)
    {
        names.push_back("inefficient");
    }
    if ((bits & (1u << static_cast<uint8_t>(AnalysisType::OOM_ANALYSIS))) != 0)
    {
        names.push_back("oom");
    }
    if ((bits & (1u << static_cast<uint8_t>(AnalysisType::HOST_LEAK_ANALYSIS))) != 0)
    {
        names.push_back("host-leaks");
    }
    if (names.empty())
    {
        return "none";
    }
    std::string joined;
    for (size_t i = 0; i < names.size(); ++i)
    {
        if (i > 0)
        {
            joined += ",";
        }
        joined += names[i];
    }
    return joined;
}

// 无符号整数严格解析(空白/符号/溢出/非法字符均拒绝);失败返回false
bool ParseUInt64(const std::string& text, uint64_t& out)
{
    if (text.empty())
    {
        return false;
    }
    for (char c : text)
    {
        if (!std::isdigit(static_cast<unsigned char>(c)))
        {
            return false;
        }
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0')
    {
        return false;
    }
    out = static_cast<uint64_t>(v);
    return true;
}

// --call-stack值解析:"c[:depth],python[:depth]"逗号组合(兼容中英文逗号),
// 缺省depth=50(与DEFAULT_CALL_STACK_DEPTH一致);"none"关闭;深度范围1..1000
// (与CLI一致);空值("c:"/"python:")报错;非法返回false
bool ParseCallStack(const std::string& text, bool& enableC, uint32_t& depthC, bool& enablePy, uint32_t& depthPy)
{
    if (text == "none")
    {
        enableC = false;
        enablePy = false;
        return true;
    }
    bool anyValid = false;
    const std::vector<std::string> tokens = Utility::SplitString(text, "，,");
    for (const std::string& token : tokens)
    {
        if (token == "c")
        {
            enableC = true;
            depthC = DEFAULT_CALL_STACK_DEPTH;
            anyValid = true;
        }
        else if (token.rfind("c:", 0) == 0)
        {
            uint64_t depth = 0;
            if (token.size() == 2 || !ParseUInt64(token.substr(2), depth) || depth < 1 || depth > 1000)
            {
                return false;  // 空值/超界:报错而非静默回落默认深度
            }
            enableC = true;
            depthC = static_cast<uint32_t>(depth);
            anyValid = true;
        }
        else if (token == "python")
        {
            enablePy = true;
            depthPy = DEFAULT_CALL_STACK_DEPTH;
            anyValid = true;
        }
        else if (token.rfind("python:", 0) == 0)
        {
            uint64_t depth = 0;
            if (token.size() == 7 || !ParseUInt64(token.substr(7), depth) || depth < 1 || depth > 1000)
            {
                return false;
            }
            enablePy = true;
            depthPy = static_cast<uint32_t>(depth);
            anyValid = true;
        }
        else
        {
            return false;
        }
    }
    return anyValid;
}
}  // namespace

ControlCommandHandler& ControlCommandHandler::GetInstance()
{
    static ControlCommandHandler handler;
    return handler;
}

void ControlCommandHandler::EventHandle(std::shared_ptr<EventBase>& event, MemoryState* state)
{
    HandleDispatch(event, state);
}

void ControlCommandHandler::HandleDispatch(std::shared_ptr<EventBase>& event, MemoryState* state)
{
    (void)state;
    std::shared_ptr<ControlEvent> ev = std::dynamic_pointer_cast<ControlEvent>(event);
    if (ev == nullptr)
    {
        return;
    }
    // 派发内仅白名单校验(不执行命令):非法词直接回执,合法词留待Execute执行
    const std::vector<std::string> tokens = Utility::SplitString(ev->cmd, " ");
    if (!ControlProtocol::IsValidControlWord(tokens))
    {
        ev->output = "invalid control word: " + ev->cmd;
        ev->ok = false;  // 白名单校验失败:单发退出码1
    }
}

void ControlCommandHandler::Execute(std::shared_ptr<ControlEvent>& ev)
{
    if (ev == nullptr)
    {
        return;
    }
    const std::vector<std::string> tokens = Utility::SplitString(ev->cmd, " ");
    if (!ControlProtocol::IsValidControlWord(tokens))
    {
        // 防御性兜底(与HandleDispatch独立校验):未过白名单不执行
        if (ev->output.empty())
        {
            ev->output = "invalid control word: " + ev->cmd;
            ev->ok = false;
        }
        return;
    }
    const std::string& head = tokens[0];
    if (head == "start")
    {
        DoStart(ev);
    }
    else if (head == "stop")
    {
        DoStop(ev);
    }
    else if (head == "step")
    {
        DoStep(ev);
    }
    else if (head == "display")
    {
        if (tokens[1] == "hook")
        {
            DoDisplayHook(ev);
        }
        else if (tokens[1] == "analyzer")
        {
            DoDisplayAnalyzer(ev);
        }
        else if (tokens[1] == "config")
        {
            DoDisplayConfig(ev);
        }
        else if (tokens.size() >= 3 && tokens[1] == "memory" && tokens[2] == "summary")
        {
            DoDisplayMemorySummary(ev);
        }
        else if (tokens.size() >= 3 && tokens[1] == "memory" && tokens[2] == "block")
        {
            DoDisplayMemoryBlock(ev);
        }
        else if (tokens.size() >= 3 && tokens[1] == "host_leak" && tokens[2] == "summary")
        {
            DoDisplayHostLeakSummary(ev);
        }
        else
        {
            // 白名单允许但缺下级词(display memory/display host_leak):给可操作提示
            ev->output =
                "invalid control word: " + ev->cmd + " (display memory: summary|block; display host_leak: summary)";
            ev->ok = false;
        }
    }
    else if (head == "set")
    {
        DoSetConfig(ev);
    }
}

void ControlCommandHandler::DoStart(std::shared_ptr<ControlEvent>& ev)
{
    if (EventTraceManager::Instance().IsTracingEnabled())
    {
        ev->output = "tracing already in progress";
        return;
    }
    ConfigManager::Instance().InitStartConfig();
    ev->output = "tracing started";
}

void ControlCommandHandler::DoStop(std::shared_ptr<ControlEvent>& ev)
{
    if (!EventTraceManager::Instance().IsTracingEnabled())
    {
        ev->output = "not in tracing";
        return;
    }
    EventTraceManager::Instance().SetTraceStatus(EventTraceStatus::NOT_IN_TRACING);
    EventTraceManager::Instance().CleanUpEventTraceManager();
    ev->output = "tracing stopped";
}

void ControlCommandHandler::DoStep(std::shared_ptr<ControlEvent>& ev)
{
    if (!EventTraceManager::Instance().IsTracingEnabled())
    {
        ev->output = "step ignored: not in tracing (start first)";
        return;
    }
    EventReport& report = EventReport::Instance(MemScopeCommType::SHARED_MEMORY);
    if (!report.ReportPyStepRecord())
    {
        ev->output = "step failed: ReportPyStepRecord error";
        ev->ok = false;
        return;
    }
    ev->output = "step " + std::to_string(report.GetPyStepId()) + " recorded";
}

void ControlCommandHandler::DoDisplayHook(std::shared_ptr<ControlEvent>& ev)
{
    std::ostringstream out;
    out << "Host hook: " << (IsSoLoaded(HOST_HOOK_SO_NAME) ? "loaded" : "not loaded") << "\n";
    out << "NPU hooks: " << (IsNpuHookLoaded() ? "loaded" : "not loaded");
    ev->output = out.str();
}

void ControlCommandHandler::DoDisplayAnalyzer(std::shared_ptr<ControlEvent>& ev)
{
    const std::vector<std::string> names = EventDispatcher::GetInstance().GetSubscriberNames();
    std::ostringstream out;
    out << "Analyzers: ";
    bool first = true;
    for (const std::string& name : names)
    {
        if (name == GetName())
        {
            continue;  // 控制通道自身不进展示
        }
        if (!first)
        {
            out << ",";
        }
        out << name;
        first = false;
    }
    ev->output = out.str();
}

void ControlCommandHandler::DoDisplayConfig(std::shared_ptr<ControlEvent>& ev)
{
    const Config& cfg = ConfigManager::Instance().GetConfig();
    std::ostringstream out;
    out << "analysis: " << AnalysisTypeToNames(cfg.analysisType) << "\n";
    out << "host_leak_mode: " << (cfg.hostLeakMode == static_cast<uint8_t>(HostLeakMode::SUMMARY) ? "summary" : "event")
        << "\n";
    out << "block_size_threshold: " << cfg.blockSizeThreshold << "\n";
    out << "sample_rate: " << cfg.sampleRate << "\n";
    out << "collect_mode: "
        << (cfg.collectMode == static_cast<uint8_t>(CollectMode::IMMEDIATE) ? "immediate" : "deferred") << "\n";
    std::string callStack;
    if (cfg.enableCStack)
    {
        callStack += "c:" + std::to_string(cfg.cStackDepth);
    }
    if (cfg.enablePyStack)
    {
        if (!callStack.empty())
        {
            callStack += ",";
        }
        callStack += "python:" + std::to_string(cfg.pyStackDepth);
    }
    if (callStack.empty())
    {
        callStack = "none";
    }
    out << "call_stack: " << callStack << "\n";
    out << "output_dir: " << cfg.outputDir << "\n";
    out << "log_level: " << static_cast<uint32_t>(cfg.logLevel);
    ev->output = out.str();
}

void ControlCommandHandler::DoDisplayMemorySummary(std::shared_ptr<ControlEvent>& ev)
{
    MemoryStateManager& msm = MemoryStateManager::GetInstance();
    std::ostringstream out;

    // 卡级行:hal有数据∪device/process缓存非-1(GetUsedDeviceList语义)
    const std::vector<int32_t> devices = msm.GetUsedDeviceList();
    if (devices.empty())
    {
        out << "(no device data collected yet)\n";
    }
    for (int32_t dev : devices)
    {
        out << "Device " << dev << ":";
        const int64_t deviceUsed = msm.GetDeviceUsed(dev);
        out << " device_used=" << (deviceUsed < 0 ? "unknown" : FormatSizeAbbrev(deviceUsed));
        const int64_t processUsed = msm.GetProcessUsed(dev);
        out << " process_used=" << (processUsed < 0 ? "unknown" : FormatSizeAbbrev(processUsed));
        out << " hal_current=" << FormatSizeAbbrev(msm.GetHalUsed(dev));
        out << " hal_peak=" << FormatSizeAbbrev(msm.GetHalPeak(dev)) << "\n";
    }

    // 锁页/CPU tensor(无数据不展示:无记录时current/peak均为0)
    if (msm.GetHostPinnedPeak() > 0)
    {
        out << "Host pinned: current=" << FormatSizeAbbrev(msm.GetHostPinnedUsed())
            << " peak=" << FormatSizeAbbrev(msm.GetHostPinnedPeak()) << "\n";
    }
    if (msm.GetHostTensorPeak() > 0)
    {
        out << "Host tensor: current=" << FormatSizeAbbrev(msm.GetHostTensorUsed())
            << " peak=" << FormatSizeAbbrev(msm.GetHostTensorPeak()) << "\n";
    }

    // 其余池(PTA_CACHING/PTA_WORKSPACE/MINDSPORE/ATB):逐池遍历16卡取有数据设备
    const std::vector<PoolType> genericPools = {PoolType::PTA_CACHING, PoolType::PTA_WORKSPACE, PoolType::MINDSPORE,
                                                PoolType::ATB};
    for (PoolType pool : genericPools)
    {
        bool printed = false;
        for (int32_t dev = 0; dev < 16; ++dev)
        {
            const int64_t current = msm.GetPoolCurrent(pool, dev);
            const int64_t peak = msm.GetPoolPeak(pool, dev);
            if (current == 0 && peak == 0)
            {
                continue;
            }
            if (!printed)
            {
                out << "Pool ";
                out << (pool == PoolType::PTA_CACHING     ? "pta"
                        : pool == PoolType::PTA_WORKSPACE ? "pta_workspace"
                        : pool == PoolType::MINDSPORE     ? "mindspore"
                                                          : "atb");
                out << ":\n";
                printed = true;
            }
            out << "  Device " << dev << ": current=" << FormatSizeAbbrev(current) << " peak=" << FormatSizeAbbrev(peak)
                << "\n";
        }
    }
    ev->output = out.str();
}

void ControlCommandHandler::DoDisplayMemoryBlock(std::shared_ptr<ControlEvent>& ev)
{
    // param解析: --pool <name> [--TOPN <N>] (--TOPN 1~100,默认10;不指定池=全部池)
    std::string poolName;
    uint64_t topN = 10;
    const std::vector<std::string> args = Utility::SplitString(ev->param, " \t");
    for (size_t i = 0; i < args.size(); ++i)
    {
        const std::string& arg = args[i];
        std::string key = arg;
        std::string value;
        const size_t eq = arg.find('=');
        if (eq != std::string::npos)
        {
            key = arg.substr(0, eq);
            value = arg.substr(eq + 1);
        }
        else if (i + 1 < args.size())
        {
            value = args[++i];
        }
        if (key == "--pool")
        {
            if (value.empty() || value.rfind("--", 0) == 0)
            {
                // 值以--开头=下一个选项被误当值(如"--pool --TOPN 20")
                ev->output = "missing value for --pool (option value cannot start with --)";
                ev->ok = false;
                return;
            }
            poolName = value;
        }
        else if (key == "--TOPN")
        {
            uint64_t n = 0;
            if (!ParseUInt64(value, n) || n < 1 || n > 100)
            {
                ev->output = "invalid --TOPN value (range 1..100): " + value;
                ev->ok = false;
                return;
            }
            topN = n;
        }
        else
        {
            ev->output = "invalid option: " + key + " (expect --pool <name> [--TOPN <N>])";
            ev->ok = false;
            return;
        }
    }
    if (!poolName.empty() && PoolNameToType(poolName) == PoolType::INVALID)
    {
        ev->output = "invalid pool name: " + poolName + " (expect host/hal/pta/pta_workspace/atb/mindspore)";
        ev->ok = false;
        return;
    }

    LiveBlockFilter filter;
    if (!poolName.empty())
    {
        filter.poolTypes.push_back(PoolNameToType(poolName));
    }
    std::vector<LiveBlockInfo> blocks = MemoryStateManager::GetInstance().QueryLiveBlocks(filter);
    std::sort(blocks.begin(), blocks.end(),
              [](const LiveBlockInfo& a, const LiveBlockInfo& b) { return a.size > b.size; });

    std::ostringstream out;
    out << "Top " << topN << " live blocks by size (of " << blocks.size()
        << " live, pool=" << (poolName.empty() ? "all" : poolName) << "):\n";
    out << "addr\t\t\tsize\talloc_ts\tallocation_id\n";
    const size_t count = std::min(static_cast<size_t>(topN), blocks.size());
    for (size_t i = 0; i < count; ++i)
    {
        char addrBuf[32];
        std::snprintf(addrBuf, sizeof(addrBuf), "0x%016llx", static_cast<unsigned long long>(blocks[i].addr));
        out << addrBuf << "\t" << blocks[i].size << "\t" << blocks[i].allocTimestamp << "\t" << blocks[i].allocationId
            << "\n";
    }
    ev->output = out.str();
}

void ControlCommandHandler::DoDisplayHostLeakSummary(std::shared_ptr<ControlEvent>& ev)
{
    // 窗口开着→钩子中间快照(不闭窗)渲染;关着→提示(E阶段完整实现)
    std::string text;
    if (HostLeakAnalyzer::GetInstance().QueryInterimOverview(text))
    {
        ev->output = text;
    }
    else
    {
        ev->output = text.empty() ? "no active window" : text;
    }
}

void ControlCommandHandler::DoSetConfig(std::shared_ptr<ControlEvent>& ev)
{
    // stop-only门控:窗口开着的配置变更会与在途采集状态冲突,要求先stop
    if (EventTraceManager::Instance().IsTracingEnabled())
    {
        ev->output = "set config rejected: tracing in progress (stop first)";
        ev->ok = false;
        return;
    }
    Config cfg = ConfigManager::Instance().GetConfig();  // 全量取当前,只改子集字段

    const std::vector<std::string> args = Utility::SplitString(ev->param, " \t");
    if (args.empty())
    {
        ev->output =
            "set config: no options given (--analysis/--host-leak-mode/--block-size-threshold/"
            "--call-stack)";
        ev->ok = false;
        return;
    }
    for (size_t i = 0; i < args.size(); ++i)
    {
        const std::string& arg = args[i];
        if (arg.rfind("--", 0) != 0)
        {
            ev->output = "invalid argument: " + arg + " (expect --key[=value])";
            ev->ok = false;
            return;
        }
        std::string key = arg;
        std::string value;
        const size_t eq = arg.find('=');
        if (eq != std::string::npos)
        {
            key = arg.substr(0, eq);
            value = arg.substr(eq + 1);
        }
        else if (i + 1 < args.size())
        {
            value = args[++i];
        }
        else
        {
            ev->output = "missing value for " + key;
            ev->ok = false;
            return;
        }

        if (key == "--analysis")
        {
            // 与python config()同源解析:host-leaks与npu分析项互斥;oom[:K]解析K
            bool hasHostLeaks = false;
            bool hasNpuAnalysis = false;
            const std::vector<std::string> tokens = Utility::SplitString(value, "，,");
            uint8_t bits = 0;
            for (const std::string& token : tokens)
            {
                if (token == "none")
                {
                    continue;
                }
                if (token == "host-leaks")
                {
                    hasHostLeaks = true;
                    bits |= (1u << static_cast<uint8_t>(AnalysisType::HOST_LEAK_ANALYSIS));
                }
                else if (token == "leaks")
                {
                    hasNpuAnalysis = true;
                    bits |= (1u << static_cast<uint8_t>(AnalysisType::LEAKS_ANALYSIS));
                }
                else if (token == "decompose")
                {
                    hasNpuAnalysis = true;
                    bits |= (1u << static_cast<uint8_t>(AnalysisType::DECOMPOSE_ANALYSIS));
                }
                else if (token == "inefficient")
                {
                    hasNpuAnalysis = true;
                    bits |= (1u << static_cast<uint8_t>(AnalysisType::INEFFICIENCY_ANALYSIS));
                }
                else if (token == "oom")
                {
                    hasNpuAnalysis = true;
                    bits |= (1u << static_cast<uint8_t>(AnalysisType::OOM_ANALYSIS));
                }
                else if (token.rfind("oom:", 0) == 0)
                {
                    hasNpuAnalysis = true;
                    bits |= (1u << static_cast<uint8_t>(AnalysisType::OOM_ANALYSIS));
                    uint64_t k = 0;
                    // 空值("oom:")与超界(1..1000)均报错,不静默沿用旧K
                    if (token.size() == 4 || !ParseUInt64(token.substr(4), k) || k < 1 || k > 1000)
                    {
                        ev->output = "invalid oom top-K (range 1..1000): " + token;
                        ev->ok = false;
                        return;
                    }
                    cfg.oomTopK = static_cast<uint16_t>(k);
                }
                else
                {
                    ev->output = "invalid analysis token: " + token +
                                 " (expect leaks/decompose/inefficient/oom[:K]/host-leaks/none)";
                    ev->ok = false;
                    return;
                }
            }
            if (hasHostLeaks && hasNpuAnalysis)
            {
                ev->output =
                    "invalid analysis: host-leaks is mutually exclusive with "
                    "leaks/decompose/inefficient/oom";
                ev->ok = false;
                return;
            }
            cfg.analysisType = bits;
        }
        else if (key == "--host-leak-mode")
        {
            if (value == "summary")
            {
                cfg.hostLeakMode = static_cast<uint8_t>(HostLeakMode::SUMMARY);
            }
            else if (value == "event")
            {
                cfg.hostLeakMode = static_cast<uint8_t>(HostLeakMode::EVENT);
            }
            else
            {
                ev->output = "invalid --host-leak-mode value: " + value + " (expect event/summary)";
                ev->ok = false;
                return;
            }
        }
        else if (key == "--block-size-threshold")
        {
            uint64_t threshold = 0;
            if (!ParseUInt64(value, threshold))
            {
                ev->output = "invalid " + key + " value: " + value;
                ev->ok = false;
                return;
            }
            cfg.blockSizeThreshold = threshold;
        }
        else if (key == "--call-stack")
        {
            if (!ParseCallStack(value, cfg.enableCStack, cfg.cStackDepth, cfg.enablePyStack, cfg.pyStackDepth))
            {
                ev->output = "invalid --call-stack value: " + value + " (expect c[:depth],python[:depth])";
                ev->ok = false;
                return;
            }
        }
        else
        {
            ev->output = "unknown option: " + key;
            ev->ok = false;
            return;
        }
    }

    ConfigManager::Instance().SetConfig(cfg);
    // 分析器订阅随analysisType变化(与python config()路径对齐:
    // EventTraceManager::SetConfig内部不触发UpdateAnalysisType)
    EventReport::Instance(MemScopeCommType::SHARED_MEMORY).UpdateAnalysisType();
    ev->output = "config updated (applied at next start)";
}

}  // namespace MemScope
