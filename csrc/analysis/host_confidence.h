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

#ifndef HOST_CONFIDENCE_H
#define HOST_CONFIDENCE_H

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace MemScope
{

/*
 * host堆内存泄漏点置信度分析(纯算法模块,供HostLeakAnalyzer消费,进程内不导出)
 *
 * 输入(闭窗快照,无事件流依赖): 节拍快照序列(每拍一行=beat/liveBytes/liveCount,
 * 写满折半折叠——行值=覆盖拍字节和,按相邻行拍差折算每拍均值入回归);
 * 栈级闭窗统计(申请/未释放/生命周期三字段); 窗口边界。
 *
 * 输出: 候选栈集(=窗口内unfreed>0的栈)逐栈五因子/LSI/常驻分类/降级标注,
 * 按LSI降序。常驻分类仅影响呈现(过滤出疑似榜),不影响LSI计算。
 *
 * 五因子(权重0.26/0.21/0.12/0.12/0.29为设计定值):
 *   G=clip(β/μA,0,1), μA=allocBytes/(K+1), K=末拍号, β=回归斜率(预热段W后)
 *   R=unfreedBytes/allocBytes(闭窗派生)
 *   S=clip((meanLiveAge−meanFreedLife)/(meanFreedLife+ε),0,1), 样本不足=0.5中性
 *   P=clip((β_recent−β_base)/(|β_recent|+|β_base|+ε),0,1), Q=末25%或末10拍取大
 *   E=max(clip(log10(1+count)/2), clip(log10(1+bytes/1MiB)/2.8))
 *   LSI=100·(0.26G+0.21R+0.12S+0.12P+0.29E)
 *
 * 常驻基线 ⟺ [G<0.05 且 最新分配时刻≤窗口起始+0.2×时长](early)
 *          或 [unfreed/alloc≤0.3 且 尾部净增速≤0.05](turnover)
 *
 * 数据不足降级(不阻塞,LSI由可得因子支撑): no_series/insufficient_series/
 * series_evicted/no_warmup_thread; 无序列时G/P=0中性
 */

// 设计定值,编译期内置常量(无外部输入,权重和经静态断言校验)
struct ConfidenceConfig
{
    double weights[5];          // {wg, wr, ws, wp, we}, 和=1
    uint32_t warmupBeats;       // 回归预热段拍数W
    uint32_t minPoints;         // 回归最少点数MIN_SERIES_POINTS
    double gResident;           // 常驻G阈值
    double residentTailFrac;    // 常驻early判据的窗口起始占比
    double turnoverResident;    // 周转常驻占比阈值
    double suspectedThreshold;  // suspected_leak状态LSI阈值
};

constexpr ConfidenceConfig kConfidenceConfig = {{0.26, 0.21, 0.12, 0.12, 0.29}, 5, 3, 0.05, 0.2, 0.3, 60.0};

// 节拍序列行(钩子侧折半折叠后,beat=首行拍号,值=覆盖拍字节和/计数和)
struct SeriesPoint
{
    uint32_t beat = 0;       // 拍号(0起,经seriesStartTsNs/beatIntervalNs映射到时间轴)
    uint64_t liveBytes = 0;  // 覆盖拍存活字节和(未折叠=该拍快照)
    uint32_t liveCount = 0;  // 覆盖拍存活块数和
};

// 单栈序列(分析器侧聚合;evicted=槽被驱逐——序列止于驱逐拍,按无序列中性处理)
struct StackSeries
{
    uint64_t stackId = 0;
    std::vector<SeriesPoint> points;
    bool evicted = false;
};

// 栈级闭窗统计(dump_stack_stats扩展字段;freedCount=allocCount−unfreedCount派生)
struct StackCloseStats
{
    uint64_t allocCount = 0;
    uint64_t allocBytes = 0;
    uint64_t unfreedCount = 0;
    uint64_t unfreedBytes = 0;
    uint64_t maxAllocTsNs = 0;        // 未释放块最新分配时刻(常驻early判据)
    uint64_t freedLifetimeSumNs = 0;  // 已释放块寿命和(free路径锁内累加)
    uint64_t liveAgeSumNs = 0;        // 未释放块年龄和(闭窗遍历顺带统计)
};

// 状态分类: 疑似泄漏/增长观察/常驻基线
enum ConfidenceStatus : uint8_t
{
    CONF_STATUS_SUSPECTED = 0,
    CONF_STATUS_GROWTH_WATCH = 1,
    CONF_STATUS_RESIDENT = 2,
};

// 降级标注: 无序列/槽被驱逐/序列过短/全窗无预热线程
enum ConfidenceDegraded : uint8_t
{
    CONF_DEGRADED_NONE = 0,
    CONF_DEGRADED_NO_SERIES = 1,
    CONF_DEGRADED_SERIES_EVICTED = 2,
    CONF_DEGRADED_INSUFFICIENT_SERIES = 3,
    CONF_DEGRADED_NO_WARMUP_THREAD = 4,
};

// 常驻判据来源: 早期放置(条件(1)(2))或周转形态(条件(4)(5))
enum ConfidenceResidentCriterion : uint8_t
{
    CONF_RESIDENT_NONE = 0,
    CONF_RESIDENT_EARLY = 1,
    CONF_RESIDENT_TURNOVER = 2,
};

struct PerStackResult
{
    uint64_t stackId = 0;
    uint8_t status = CONF_STATUS_GROWTH_WATCH;  // 常驻栈=RESIDENT,余按LSI分档
    double lsi = 0;                             // 100·加权和(未取整,报告四舍五入)
    double g = 0;
    double r = 0;
    double s = 0;
    double p = 0;
    double e = 0;
    double growthRate = 0;      // 回归斜率(B/beat)
    bool growthValid = false;   // 回归点数达标(否则growthRate/seriesPoints无效)
    uint64_t seriesPoints = 0;  // 参与回归的有效拍数
    uint8_t degraded = CONF_DEGRADED_NONE;
    bool turnoverMark = false;                       // 满足占比判据但尾部仍涨: 留疑似榜追加/turnover
    uint8_t residentCriterion = CONF_RESIDENT_NONE;  // 常驻判据来源(early/turnover)
};

class HostConfidence
{
   public:
    // 闭窗计算: 喂节拍序列、栈级统计、窗口边界,产出候选栈集结果(按LSI降序)。
    // warmupThreadFailed=本窗口预热线程创建失败(全窗无序列,统一no_warmup_thread降级)
    std::vector<PerStackResult> Compute(const std::vector<StackSeries>& series,
                                        const std::unordered_map<uint64_t, StackCloseStats>& stackStats,
                                        uint64_t windowStartTs, uint64_t windowEndTs, bool warmupThreadFailed);

    // 拍号→窗口时间轴映射(seriesStartTs为拍0锚点,CLOCK_REALTIME与块allocTs同钟)
    static uint64_t BeatToTime(uint64_t beat, uint64_t seriesStartTs, uint64_t beatIntervalNs);
};

}  // namespace MemScope

#endif
