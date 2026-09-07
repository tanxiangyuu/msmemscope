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

#include "host_confidence.h"

#include <algorithm>
#include <cmath>

namespace MemScope
{

// ---- 因子内部常量(设计定值,见头文件模块注释) ----
namespace
{

constexpr double kMinFreedSamples = 8.0;    // S因子已释放样本门限
constexpr double kEpsLifeNs = 1.0;          // S分母防零(ns)
constexpr double kEpsGrowth = 1.0;          // P分母防零(B/beat)
constexpr uint32_t kRecentMinBeats = 10;    // P尾部段最小拍数(Q下限)
constexpr double kRecentTailFrac = 0.25;    // P尾部段占比(Q=末25%)
constexpr double kBytesPerMiB = 1048576.0;  // E因子字节维度基准

// 权重和=1编译期校验(浮点容差1e-3)
static_assert(kConfidenceConfig.weights[0] + kConfidenceConfig.weights[1] + kConfidenceConfig.weights[2] +
                          kConfidenceConfig.weights[3] + kConfidenceConfig.weights[4] >=
                      0.999 &&
                  kConfidenceConfig.weights[0] + kConfidenceConfig.weights[1] + kConfidenceConfig.weights[2] +
                          kConfidenceConfig.weights[3] + kConfidenceConfig.weights[4] <=
                      1.001,
              "confidence weights must sum to 1");

double Clip01(double v) { return (v < 0.0) ? 0.0 : ((v > 1.0) ? 1.0 : v); }

// 回归点: (x=拍号, y=行值/覆盖拍数(每拍均值), w=覆盖拍数)。
// 相邻行拍差=上一行覆盖拍数,末行按1拍(窗口尾近似)。折叠行与未折叠行统一。
struct RegPoint
{
    double x;
    double y;
    double w;
};

// 加权最小二乘斜率: β = Σw(x−x̄)(y−ȳ)/Σw(x−x̄)², x̄/ȳ为加权均值。
// 点数<MIN_SERIES_POINTS或全部同x(退化)返回false
bool LeastSquaresSlope(const std::vector<RegPoint>& pts, double& slope)
{
    const uint32_t minPoints = kConfidenceConfig.minPoints;
    if (pts.size() < minPoints)
    {
        return false;
    }
    double sx = 0.0;
    double sy = 0.0;
    double sw = 0.0;
    for (const auto& p : pts)
    {
        sx += p.w * p.x;
        sy += p.w * p.y;
        sw += p.w;
    }
    if (sw <= 0.0)
    {
        return false;
    }
    const double mx = sx / sw;
    const double my = sy / sw;
    double num = 0.0;
    double den = 0.0;
    for (const auto& p : pts)
    {
        const double dx = p.x - mx;
        num += p.w * dx * (p.y - my);
        den += p.w * dx * dx;
    }
    if (den <= 0.0)
    {
        return false;  // 全部同x(单拍覆盖),斜率不可辨识
    }
    slope = num / den;
    return true;
}

// 序列行→回归点集: 仅拍号≥预热段W的行入回归;排序/去重(同拍保留末行)由调用方完成。
// 说明: 行值语义=覆盖拍字节和,相邻行拍差折算覆盖拍数,末行按1拍近似——精确覆盖
// 无丢拍记录;top-K掉出再回(存在行间隔)时,间隔前一行的覆盖拍数被高估、y被低估,
// 属记录稀疏导致的近似(单一间隔点,影响有界)。折叠行(值=多拍和)同式折算均值。
void BuildRegPoints(const std::vector<SeriesPoint>& rows, uint32_t warmupBeats, std::vector<RegPoint>& out)
{
    out.clear();
    out.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i)
    {
        if (rows[i].beat < warmupBeats)
        {
            continue;
        }
        const uint64_t nextBeat = (i + 1 < rows.size()) ? rows[i + 1].beat : rows[i].beat;
        uint64_t span = nextBeat - rows[i].beat;  // 拍差≥0(已排序去重),末行span=0→按1拍
        if (span < 1)
        {
            span = 1;
        }
        RegPoint p;
        p.x = static_cast<double>(rows[i].beat);
        p.y = static_cast<double>(rows[i].liveBytes) / static_cast<double>(span);
        p.w = static_cast<double>(span);
        out.push_back(p);
    }
}

}  // namespace

uint64_t HostConfidence::BeatToTime(uint64_t beat, uint64_t seriesStartTs, uint64_t beatIntervalNs)
{
    return seriesStartTs + beat * beatIntervalNs;
}

std::vector<PerStackResult> HostConfidence::Compute(const std::vector<StackSeries>& series,
                                                    const std::unordered_map<uint64_t, StackCloseStats>& stackStats,
                                                    uint64_t windowStartTs, uint64_t windowEndTs,
                                                    bool warmupThreadFailed)
{
    // 序列按栈索引,候选栈(=窗口内unfreed>0)逐个计算
    std::unordered_map<uint64_t, const StackSeries*> seriesById;
    seriesById.reserve(series.size());
    for (const auto& s : series)
    {
        if (!s.points.empty())
        {
            seriesById.emplace(s.stackId, &s);
        }
    }

    const uint32_t warmupBeats = kConfidenceConfig.warmupBeats;
    const double suspectedThreshold = kConfidenceConfig.suspectedThreshold;

    std::vector<PerStackResult> results;
    results.reserve(stackStats.size());

    for (const auto& kv : stackStats)
    {
        const uint64_t stackId = kv.first;
        const StackCloseStats& st = kv.second;
        // 候选集=窗口内unfreed>0的栈(与TOP N列表同源);无未释放者不产出结果
        if (st.unfreedCount == 0)
        {
            continue;
        }

        PerStackResult r;
        r.stackId = stackId;

        // 尾部段斜率β_recent(常驻turnover判据); 无序列/尾部段数据不足时保持0/无效
        double betaRecent = 0.0;
        bool recentValid = false;

        // 降级优先级: 全窗无预热线程 > 无序列 > 槽被驱逐 > 序列过短; 无序列
        // 或槽被驱逐时G/P=0中性,LSI由R/S/E支撑
        if (warmupThreadFailed)
        {
            r.degraded = CONF_DEGRADED_NO_WARMUP_THREAD;
        }
        else
        {
            const auto it = seriesById.find(stackId);
            if (it == seriesById.end())
            {
                r.degraded = CONF_DEGRADED_NO_SERIES;
            }
            else if (it->second->evicted)
            {
                // 槽被驱逐: 序列止于驱逐拍(覆盖不完整),按无序列中性处理,仅标注
                r.degraded = CONF_DEGRADED_SERIES_EVICTED;
            }
            else
            {
                // 行预处理: 按拍升序,同拍保留末行(容忍乱序/重复注入,后行胜)
                std::vector<SeriesPoint> rows = it->second->points;
                std::sort(rows.begin(), rows.end(),
                          [](const SeriesPoint& a, const SeriesPoint& b) { return a.beat < b.beat; });
                std::vector<SeriesPoint> deduped;
                deduped.reserve(rows.size());
                for (const auto& row : rows)
                {
                    if (!deduped.empty() && deduped.back().beat == row.beat)
                    {
                        deduped.back() = row;  // 同拍保留末行
                    }
                    else
                    {
                        deduped.push_back(row);
                    }
                }
                rows.swap(deduped);

                std::vector<RegPoint> regPoints;
                BuildRegPoints(rows, warmupBeats, regPoints);
                const bool regValid = LeastSquaresSlope(regPoints, r.growthRate);
                r.growthValid = regValid;
                if (regValid)
                {
                    // 参与回归的有效拍数=回归窗口内各行覆盖拍和(无丢拍时=末拍−预热拍+1)
                    r.seriesPoints = 0;
                    for (const auto& p : regPoints)
                    {
                        r.seriesPoints += static_cast<uint64_t>(p.w);
                    }
                }
                else
                {
                    r.degraded = CONF_DEGRADED_INSUFFICIENT_SERIES;
                }

                // 增长因子G=clip(β/μA,0,1), μA=allocBytes/(K+1), K=末拍号;
                // 回归不可得或申请为0→0中性
                const uint64_t kLastBeat = rows.back().beat;
                if (regValid && st.allocBytes > 0)
                {
                    const double muA = static_cast<double>(st.allocBytes) / static_cast<double>(kLastBeat + 1);
                    r.g = Clip01(r.growthRate / muA);
                }
                else
                {
                    r.g = 0.0;
                }

                // 模式因子P=clip((β_recent−β_base)/(|β_recent|+|β_base|+ε),0,1);
                // Q=末25%或末10拍取大,段=回归窗口内以beat≤K−Q分界;
                // 任一段点数不足→P=0中性(β_recent按0,常驻turnover判据视为满足)
                const double kLast = static_cast<double>(kLastBeat);
                uint64_t qBeats = (kLastBeat + 1 > warmupBeats) ? (kLastBeat + 1 - warmupBeats) : 0;
                qBeats = static_cast<uint64_t>(qBeats * kRecentTailFrac);
                if (qBeats < kRecentMinBeats)
                {
                    qBeats = kRecentMinBeats;
                }
                const double splitBeat = kLast - static_cast<double>(qBeats);
                std::vector<RegPoint> basePts;
                std::vector<RegPoint> recentPts;
                basePts.reserve(regPoints.size());
                recentPts.reserve(regPoints.size());
                for (const auto& p : regPoints)
                {
                    if (p.x <= splitBeat)
                    {
                        basePts.push_back(p);
                    }
                    else
                    {
                        recentPts.push_back(p);
                    }
                }
                double betaBase = 0.0;
                const bool baseValid = LeastSquaresSlope(basePts, betaBase);
                recentValid = LeastSquaresSlope(recentPts, betaRecent);
                if (baseValid && recentValid)
                {
                    r.p = Clip01((betaRecent - betaBase) / (std::fabs(betaRecent) + std::fabs(betaBase) + kEpsGrowth));
                }
                else
                {
                    r.p = 0.0;  // 任一段数据不足→中性
                }
            }
        }

        // 释放率R=unfreed/alloc(闭窗派生, 释放=申请−未释放口径)
        r.r = (st.allocBytes > 0) ? static_cast<double>(st.unfreedBytes) / static_cast<double>(st.allocBytes) : 0.0;

        // 生命周期S=clip((meanLiveAge−meanFreedLife)/(meanFreedLife+ε),0,1)。
        // freedCount=allocCount−unfreedCount(闭窗派生); 样本<8或无寿命和→0.5中性
        // (分母寿命和为0时直除会饱和1虚高,同样按中性)
        const uint64_t freedCount = (st.allocCount > st.unfreedCount) ? (st.allocCount - st.unfreedCount) : 0;
        if (freedCount >= static_cast<uint64_t>(kMinFreedSamples) && st.freedLifetimeSumNs > 0 && st.unfreedCount > 0)
        {
            const double meanLiveAge = static_cast<double>(st.liveAgeSumNs) / static_cast<double>(st.unfreedCount);
            const double meanFreedLife = static_cast<double>(st.freedLifetimeSumNs) / static_cast<double>(freedCount);
            r.s = Clip01((meanLiveAge - meanFreedLife) / (meanFreedLife + kEpsLifeNs));
        }
        else
        {
            r.s = 0.5;
        }

        // 规模因子E=计数/字节双维度取大
        const double eCount = Clip01(std::log10(1.0 + static_cast<double>(st.unfreedCount)) / 2.0);
        const double eBytes = Clip01(std::log10(1.0 + static_cast<double>(st.unfreedBytes) / kBytesPerMiB) / 2.8);
        r.e = std::max(eCount, eBytes);

        // LSI = 100·加权和
        const double* w = kConfidenceConfig.weights;
        r.lsi = 100.0 * (w[0] * r.g + w[1] * r.r + w[2] * r.s + w[3] * r.p + w[4] * r.e);

        // 常驻基线分类(先于LSI分档; 仅影响呈现):
        // early=[G<0.05 且 最新分配时刻≤窗口起始+0.2×时长](G=0中性同样计入);
        // turnover=[unfreed/alloc≤0.3 且 尾部净增速≤0.05](β_recent不可得按0处理)
        const double gResident = kConfidenceConfig.gResident;
        const double turnoverResident = kConfidenceConfig.turnoverResident;
        const bool cond1 = (r.g < gResident);
        const bool cond2 =
            (windowEndTs > windowStartTs) &&
            (st.maxAllocTsNs <= windowStartTs + static_cast<uint64_t>(static_cast<double>(windowEndTs - windowStartTs) *
                                                                      kConfidenceConfig.residentTailFrac));
        const bool cond4 =
            (st.allocBytes > 0) &&
            (static_cast<double>(st.unfreedBytes) / static_cast<double>(st.allocBytes) <= turnoverResident);
        const bool cond5 = !recentValid || betaRecent <= gResident;
        const bool turnoverCriterion = cond4 && cond5;
        r.turnoverMark = cond4 && !cond5;  // 占比判据满足但尾部仍涨: 留疑似榜追加/turnover
        if (cond1 && cond2)
        {
            r.residentCriterion = CONF_RESIDENT_EARLY;
            r.status = CONF_STATUS_RESIDENT;
        }
        else if (turnoverCriterion)
        {
            r.residentCriterion = CONF_RESIDENT_TURNOVER;
            r.status = CONF_STATUS_RESIDENT;
        }
        else
        {
            r.status = (r.lsi >= suspectedThreshold) ? CONF_STATUS_SUSPECTED : CONF_STATUS_GROWTH_WATCH;
        }

        results.push_back(r);
    }

    // 按LSI降序(次键: 栈号升序,保证确定性排序)
    std::sort(results.begin(), results.end(),
              [](const PerStackResult& a, const PerStackResult& b)
              {
                  if (a.lsi != b.lsi)
                  {
                      return a.lsi > b.lsi;
                  }
                  return a.stackId < b.stackId;
              });
    return results;
}

}  // namespace MemScope
