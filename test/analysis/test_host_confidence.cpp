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

/* HostConfidence单元测试(泄漏点置信度五因子/常驻分类/降级标注,纯合成序列驱动)
 * 覆盖: G/R/S/P/E五因子与LSI加权(UT-C1..C5)、常驻基线分类(UT-C6)、拍号时间轴
 * 与乱序容错(UT-C7)、数据不足降级(UT-C8)、E规模证据与锚点验收(UT-C10/A1..A4)。
 * 权重和=1的编译期静态断言在host_confidence.cpp内(断言失败即编译失败,UT-C5)。
 * 构造约定: 行=(beat, liveBytes, liveCount),未折叠(每拍一行、span=1),
 * 末行span=1,线性数据下回归斜率为精确值;统计口径unfreed>0方入候选栈集。
 */
#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <vector>

#include "host_confidence.h"

using namespace MemScope;

namespace
{

constexpr double kTol = 1e-6;              // 精确数据下的数值容差
constexpr double kFactorTol = 1e-2;        // 形态断言容差(G≈0/P≈0等)
constexpr uint64_t kBeatIntervalNs = 1000000000ULL;
constexpr uint64_t kWindowStart = 0;
constexpr uint64_t kWindowEnd = 1000 * kBeatIntervalNs;  // 窗口时长1000s

StackCloseStats MakeStats(uint64_t allocCount, uint64_t allocBytes, uint64_t unfreedCount, uint64_t unfreedBytes,
                          uint64_t maxAllocTs = 0, uint64_t freedLifeSum = 0, uint64_t liveAgeSum = 0)
{
    StackCloseStats st;
    st.allocCount = allocCount;
    st.allocBytes = allocBytes;
    st.unfreedCount = unfreedCount;
    st.unfreedBytes = unfreedBytes;
    st.maxAllocTsNs = maxAllocTs;
    st.freedLifetimeSumNs = freedLifeSum;
    st.liveAgeSumNs = liveAgeSum;
    return st;
}

// 逐拍行追加(未折叠序列构造;liveBytes=每拍末存活字节)
void AddRow(StackSeries& s, uint32_t beat, uint64_t liveBytes, uint32_t liveCount = 1)
{
    SeriesPoint p;
    p.beat = beat;
    p.liveBytes = liveBytes;
    p.liveCount = liveCount;
    s.points.push_back(p);
}

// 等拍距线性序列: beats 0..beatCount-1, liveBytes(b)=base+slope×b (b≥0)
StackSeries LinearSeries(uint64_t stackId, int64_t base, int64_t slope, int beatCount)
{
    StackSeries s;
    s.stackId = stackId;
    for (int b = 0; b < beatCount; ++b)
    {
        AddRow(s, static_cast<uint32_t>(b), static_cast<uint64_t>(base + slope * b));
    }
    return s;
}

// 台阶序列: beats 0..step-1为low, step..beatCount-1为high(台阶后平稳→G≈0)
StackSeries StepSeries(uint64_t stackId, uint64_t low, uint64_t high, int step, int beatCount)
{
    StackSeries s;
    s.stackId = stackId;
    for (int b = 0; b < beatCount; ++b)
    {
        AddRow(s, static_cast<uint32_t>(b), b < step ? low : high);
    }
    return s;
}

// 单栈计算入口(序列/统计/边界齐备)
const PerStackResult* ComputeOne(const std::vector<StackSeries>& series,
                                 const std::unordered_map<uint64_t, StackCloseStats>& stats,
                                 uint64_t windowStart = kWindowStart, uint64_t windowEnd = kWindowEnd,
                                 bool warmupFailed = false)
{
    static std::vector<PerStackResult> last;  // 结果生命周期跨断言
    last = HostConfidence().Compute(series, stats, windowStart, windowEnd, warmupFailed);
    return last.empty() ? nullptr : &last[0];
}

}  // namespace

// ---- UT-C1: 增长因子G ----
TEST(HostConfidenceTest, C1GrowthFactor)
{
    // 线性增长(每拍+1000B,不释放): β=1000, μA=allocBytes/301
    // allocBytes=301×1000(拍0..300每拍申请1000)→G=1.0
    const uint64_t stackId = 0x11;
    const int beats = 301;
    std::vector<StackSeries> series;
    series.push_back(LinearSeries(stackId, 1000, 1000, beats));
    std::unordered_map<uint64_t, StackCloseStats> stats;
    stats.emplace(stackId, MakeStats(beats, 1000ULL * beats, beats, 1000ULL * beats));  // 不释放
    const PerStackResult* r = ComputeOne(series, stats);
    ASSERT_NE(r, nullptr);
    EXPECT_NEAR(r->g, 1.0, kTol);
    EXPECT_TRUE(r->growthValid);

    // 平稳序列(常驻): β=0 → G=0
    const uint64_t flatId = 0x12;
    std::vector<StackSeries> series2;
    series2.push_back(LinearSeries(flatId, 5000, 0, beats));
    std::unordered_map<uint64_t, StackCloseStats> stats2;
    stats2.emplace(flatId, MakeStats(1, 5000, 1, 5000));
    const PerStackResult* f = ComputeOne(series2, stats2);
    ASSERT_NE(f, nullptr);
    EXPECT_NEAR(f->g, 0.0, kTol);

    // 含噪声平稳: β≈0 → G≤0.05
    const uint64_t noisyId = 0x13;
    std::vector<StackSeries> series3;
    StackSeries ns;
    ns.stackId = noisyId;
    for (int b = 0; b < beats; ++b)
    {
        const uint64_t v = 5000 + static_cast<uint64_t>((b % 7) - 3);  // 确定性±3抖动
        AddRow(ns, static_cast<uint32_t>(b), v);
    }
    series3.push_back(ns);
    std::unordered_map<uint64_t, StackCloseStats> stats3;
    stats3.emplace(noisyId, MakeStats(1, 5000, 1, 5000));
    const PerStackResult* n = ComputeOne(series3, stats3);
    ASSERT_NE(n, nullptr);
    EXPECT_GE(n->g, 0.0);
    EXPECT_LE(n->g, 0.05);

    // 预热段(W=5)排除: 前5拍巨大、之后平稳 → 回归仅覆盖W后 → G≈0
    const uint64_t warmId = 0x14;
    std::vector<StackSeries> series4;
    series4.push_back(StepSeries(warmId, 100000, 1000, 5, beats));
    std::unordered_map<uint64_t, StackCloseStats> stats4;
    stats4.emplace(warmId, MakeStats(1, 1000, 1, 1000));
    const PerStackResult* w = ComputeOne(series4, stats4);
    ASSERT_NE(w, nullptr);
    EXPECT_LE(w->g, 0.05);

    // 扩容台阶(台阶间平稳): G介于0~1且小于持续增长
    // μA=窗口内累计申请/301(200拍×1000B+101拍×100000B=10.3MB),台阶斜率~450B/拍
    // → G≈0.01(远小于持续增长的G=1.0)
    const uint64_t stepId = 0x15;
    std::vector<StackSeries> series5;
    series5.push_back(StepSeries(stepId, 1000, 100000, 200, beats));  // 台阶在200拍
    std::unordered_map<uint64_t, StackCloseStats> stats5;
    stats5.emplace(stepId, MakeStats(301, 200ULL * 1000 + 101ULL * 100000, 1, 100000));
    const PerStackResult* s = ComputeOne(series5, stats5);
    ASSERT_NE(s, nullptr);
    EXPECT_GE(s->g, 0.0);
    EXPECT_LT(s->g, 1.0);
    EXPECT_LT(s->g, 1.0 - kFactorTol);  // 小于持续增长(上例G=1.0)

    // 点数<MIN_SERIES_POINTS: 仅2拍入回归窗口 → G=0(序列过短降级)
    const uint64_t shortId = 0x16;
    std::vector<StackSeries> series6;
    StackSeries ss;
    ss.stackId = shortId;
    AddRow(ss, 5, 1000);
    AddRow(ss, 6, 2000);
    series6.push_back(ss);
    std::unordered_map<uint64_t, StackCloseStats> stats6;
    stats6.emplace(shortId, MakeStats(2, 3000, 2, 3000));
    const PerStackResult* q = ComputeOne(series6, stats6);
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(q->g, 0.0);
    EXPECT_EQ(q->degraded, CONF_DEGRADED_INSUFFICIENT_SERIES);
    EXPECT_FALSE(q->growthValid);

    // A=0 → G=0(防御分支;unfreed>0且alloc=0为异常输入)
    const uint64_t zeroAId = 0x17;
    std::vector<StackSeries> series7;
    series7.push_back(LinearSeries(zeroAId, 1000, 1000, beats));
    std::unordered_map<uint64_t, StackCloseStats> stats7;
    stats7.emplace(zeroAId, MakeStats(0, 0, 1, 1));
    const PerStackResult* z = ComputeOne(series7, stats7);
    ASSERT_NE(z, nullptr);
    EXPECT_EQ(z->g, 0.0);
}

// ---- UT-C2: 释放率因子R ----
TEST(HostConfidenceTest, C2ReleaseFactor)
{
    const int beats = 301;
    const uint64_t stackId = 0x21;
    std::vector<StackSeries> series;
    series.push_back(LinearSeries(stackId, 1000, 1000, beats));
    std::unordered_map<uint64_t, StackCloseStats> stats;
    // 纯泄漏: 申请全部未释放 → R=1
    stats.emplace(stackId, MakeStats(beats, 1000ULL * beats, beats, 1000ULL * beats));
    const PerStackResult* r = ComputeOne(series, stats);
    ASSERT_NE(r, nullptr);
    EXPECT_NEAR(r->r, 1.0, kTol);

    // 工作内存: 申请释放平衡 → R≈0; 混合 → 派生口径R=unfreed/alloc
    const uint64_t workId = 0x22;
    std::unordered_map<uint64_t, StackCloseStats> stats2;
    stats2.emplace(workId, MakeStats(100, 100000, 10, 1000));  // 90%已释放
    std::vector<StackSeries> series2;                           // 无序列(不影响R)
    const PerStackResult* w = ComputeOne(series2, stats2);
    ASSERT_NE(w, nullptr);
    EXPECT_NEAR(w->r, 0.01, kTol);  // 1000/100000
    EXPECT_EQ(w->r, 0.01);          // 派生口径与RFC-1"释放=申请−未释放"一致

    // A=0 → R=0
    const uint64_t zeroId = 0x23;
    std::unordered_map<uint64_t, StackCloseStats> stats3;
    stats3.emplace(zeroId, MakeStats(0, 0, 1, 1));
    const PerStackResult* z = ComputeOne({}, stats3);
    ASSERT_NE(z, nullptr);
    EXPECT_EQ(z->r, 0.0);
}

// ---- UT-C3: 生命周期因子S ----
TEST(HostConfidenceTest, C3LifetimeFactor)
{
    const uint64_t ns = 1000000000ULL;  // 1s
    const uint64_t stackId = 0x31;
    // 未释放块平均年龄100s >> 已释放块平均寿命10s → S≈1
    std::unordered_map<uint64_t, StackCloseStats> stats;
    stats.emplace(stackId, MakeStats(18, 1800, 10, 1000, 0, 8 * 10 * ns, 10 * 100 * ns));
    const PerStackResult* r = ComputeOne({}, stats);
    ASSERT_NE(r, nullptr);
    EXPECT_NEAR(r->s, 1.0, kTol);

    // 未释放块年轻(10s < 已释放寿命100s) → S≈0
    const uint64_t youngId = 0x32;
    std::unordered_map<uint64_t, StackCloseStats> stats2;
    stats2.emplace(youngId, MakeStats(18, 1800, 10, 1000, 0, 8 * 100 * ns, 10 * 10 * ns));
    const PerStackResult* y = ComputeOne({}, stats2);
    ASSERT_NE(y, nullptr);
    EXPECT_NEAR(y->s, 0.0, kTol);

    // freedCount<MIN_FREED_SAMPLES(8) → S=0.5中性
    const uint64_t fewId = 0x33;
    std::unordered_map<uint64_t, StackCloseStats> stats3;
    stats3.emplace(fewId, MakeStats(9, 900, 8, 800));  // freedCount=1
    const PerStackResult* f = ComputeOne({}, stats3);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->s, 0.5);

    // freedLifetimeSum=0(有free但零寿命,如同拍内free) → S=0.5中性(防虚高)
    const uint64_t zeroLifeId = 0x34;
    std::unordered_map<uint64_t, StackCloseStats> stats4;
    stats4.emplace(zeroLifeId, MakeStats(18, 1800, 10, 1000, 0, 0, 10 * 100 * ns));  // freedLife=0
    const PerStackResult* z = ComputeOne({}, stats4);
    ASSERT_NE(z, nullptr);
    EXPECT_EQ(z->s, 0.5);

    // 无未释放块不入候选集
    const uint64_t noneId = 0x35;
    std::unordered_map<uint64_t, StackCloseStats> stats5;
    stats5.emplace(noneId, MakeStats(100, 10000, 0, 0));
    const std::vector<PerStackResult> out =
        HostConfidence().Compute({}, stats5, kWindowStart, kWindowEnd, false);
    EXPECT_TRUE(out.empty());
}

// ---- UT-C4: 模式因子P ----
TEST(HostConfidenceTest, C4PatternFactor)
{
    const int beats = 301;
    // 斜率平稳: β_base=β_recent=100 → P=0
    const uint64_t steadyId = 0x41;
    std::vector<StackSeries> series;
    series.push_back(LinearSeries(steadyId, 100, 100, beats));
    std::unordered_map<uint64_t, StackCloseStats> stats;
    stats.emplace(steadyId, MakeStats(beats, 100ULL * beats, beats, 100ULL * beats));
    const PerStackResult* s = ComputeOne(series, stats);
    ASSERT_NE(s, nullptr);
    EXPECT_NEAR(s->p, 0.0, kTol);

    // 近期段斜率显著大于基线段 → P→1(Q=末25%: 296拍窗口→Q=74拍,
    // 基线段=5..226斜率100, 尾部段=227..300斜率10100 → P≈0.98)
    const uint64_t accelId = 0x42;
    StackSeries accel;
    accel.stackId = accelId;
    for (int b = 0; b < beats; ++b)
    {
        int64_t v = 100 * static_cast<int64_t>(b);
        if (b > 226)
        {
            v += 10000 * static_cast<int64_t>(b - 227);
        }
        AddRow(accel, static_cast<uint32_t>(b), static_cast<uint64_t>(v));
    }
    std::vector<StackSeries> series2;
    series2.push_back(accel);
    std::unordered_map<uint64_t, StackCloseStats> stats2;
    stats2.emplace(accelId, MakeStats(beats, 30000ULL * beats, beats, 30000ULL * beats));
    const PerStackResult* a = ComputeOne(series2, stats2);
    ASSERT_NE(a, nullptr);
    EXPECT_GT(a->p, 0.9);
    EXPECT_LE(a->p, 1.0);

    // 基线段点数不足: 窗口仅10拍(Q=max(10, 2)=10吞掉基线段) → P=0
    const uint64_t shortId = 0x43;
    StackSeries shortS;
    shortS.stackId = shortId;
    for (int b = 5; b <= 14; ++b)
    {
        AddRow(shortS, static_cast<uint32_t>(b), static_cast<uint64_t>(100 * b));
    }
    std::vector<StackSeries> series3;
    series3.push_back(shortS);
    std::unordered_map<uint64_t, StackCloseStats> stats3;
    stats3.emplace(shortId, MakeStats(10, 1000ULL * 10, 10, 1000ULL * 10));
    const PerStackResult* q = ComputeOne(series3, stats3);
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(q->p, 0.0);

    // 负增长(收缩): β_recent=β_base<0 → 分子0 → P=0
    const uint64_t shrinkId = 0x44;
    std::vector<StackSeries> series4;
    StackSeries shr;
    shr.stackId = shrinkId;
    for (int b = 0; b < beats; ++b)
    {
        const uint64_t v = 300000 - 1000ULL * static_cast<uint64_t>(b);
        AddRow(shr, static_cast<uint32_t>(b), v);
    }
    series4.push_back(shr);
    std::unordered_map<uint64_t, StackCloseStats> stats4;
    stats4.emplace(shrinkId, MakeStats(beats, 300000ULL * beats, beats, 300000ULL * beats));
    const PerStackResult* z = ComputeOne(series4, stats4);
    ASSERT_NE(z, nullptr);
    EXPECT_NEAR(z->p, 0.0, kTol);
}

// ---- UT-C5: LSI与权重 ----
TEST(HostConfidenceTest, C5LsiWeights)
{
    const int beats = 301;
    const uint64_t stackId = 0x51;
    std::vector<StackSeries> series;
    series.push_back(LinearSeries(stackId, 1000, 1000, beats));
    // 构造各因子可独立复算的统计: R=0.5, S=0.5(无free样本), G/P由序列
    std::unordered_map<uint64_t, StackCloseStats> stats;
    stats.emplace(stackId, MakeStats(beats, 2000ULL * beats, beats / 2, 1000ULL * beats));  // R=0.5
    const PerStackResult* r = ComputeOne(series, stats);
    ASSERT_NE(r, nullptr);
    // 手工复算: 100×(0.26G+0.21×0.5+0.12×0.5+0.12P+0.29E)
    const double expected = 100.0 * (0.26 * r->g + 0.21 * 0.5 + 0.12 * 0.5 + 0.12 * r->p + 0.29 * r->e);
    EXPECT_NEAR(r->lsi, expected, 1e-6);
    EXPECT_GE(r->lsi, 0.0);
    EXPECT_LE(r->lsi, 100.0);
    // 权重和=1的编译期静态断言在host_confidence.cpp(越界即编译失败,无法运行时构造)

    // 候选栈集LSI降序排序
    const uint64_t highId = 0x52;
    std::vector<StackSeries> series2;
    series2.push_back(LinearSeries(highId, 1000, 1000, beats));
    std::unordered_map<uint64_t, StackCloseStats> stats2;
    stats2.emplace(stackId, MakeStats(beats, 2000ULL * beats, beats / 2, 1000ULL * beats));
    stats2.emplace(highId, MakeStats(beats, 1000ULL * beats, beats, 1000ULL * beats));  // 全不释放
    const std::vector<PerStackResult> out =
        HostConfidence().Compute(series2, stats2, kWindowStart, kWindowEnd, false);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_GT(out[0].lsi, out[1].lsi);  // 全泄漏栈LSI更高,置顶
}

// ---- UT-C6: 常驻基线分类 ----
TEST(HostConfidenceTest, C6ResidentClassification)
{
    const int beats = 301;
    const uint64_t kEarlyTs = 10 * kBeatIntervalNs;   // 窗口前1%
    const uint64_t kLateTs = 990 * kBeatIntervalNs;   // 窗口尾部99%
    const uint64_t kTailTs = 250 * kBeatIntervalNs;   // 窗口25%(>0.2×时长=200s)

    // 早期分配+G≈0 → 常驻(early判据)
    const uint64_t residentId = 0x61;
    std::vector<StackSeries> series;
    series.push_back(StepSeries(residentId, 1000, 1000, 5, beats));  // G≈0
    std::unordered_map<uint64_t, StackCloseStats> stats;
    stats.emplace(residentId, MakeStats(1, 1000, 1, 1000, kEarlyTs));
    const PerStackResult* r = ComputeOne(series, stats);
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->status, CONF_STATUS_RESIDENT);
    EXPECT_EQ(r->residentCriterion, CONF_RESIDENT_EARLY);

    // 慢泄漏: G小但每拍仍在新增未释放(maxAllocTs在窗口尾部) → 条件(2)拦截,非常驻
    const uint64_t slowId = 0x62;
    std::vector<StackSeries> series2;
    series2.push_back(LinearSeries(slowId, 1000, 1, beats));  // β=1,μA大→G小
    std::unordered_map<uint64_t, StackCloseStats> stats2;
    stats2.emplace(slowId, MakeStats(beats, 1000000ULL, beats, 300, kLateTs));
    const PerStackResult* s = ComputeOne(series2, stats2);
    ASSERT_NE(s, nullptr);
    EXPECT_LT(s->g, 0.05);
    EXPECT_NE(s->status, CONF_STATUS_RESIDENT);

    // 尾部大申请(条件(2)不满足) → 非常驻
    const uint64_t lateId = 0x63;
    std::vector<StackSeries> series3;
    series3.push_back(StepSeries(lateId, 1000, 1000, 5, beats));
    std::unordered_map<uint64_t, StackCloseStats> stats3;
    stats3.emplace(lateId, MakeStats(1, 1000, 1, 1000, kTailTs));  // 250s>0.2×1000s
    const PerStackResult* l = ComputeOne(series3, stats3);
    ASSERT_NE(l, nullptr);
    EXPECT_NE(l->status, CONF_STATUS_RESIDENT);

    // 条件(2)边界: maxAllocTs恰等于0.2×时长 → 满足(≤)。stats4的栈不在series中,
    // 按无序列中性处理(G=0计入cond1)——与常驻判据"无增长证据的稳定池"设计一致
    const uint64_t edgeId = 0x64;
    std::unordered_map<uint64_t, StackCloseStats> stats4;
    stats4.emplace(edgeId, MakeStats(1, 1000, 1, 1000, 200 * kBeatIntervalNs));
    const PerStackResult* e = ComputeOne(series, stats4);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(e->status, CONF_STATUS_RESIDENT);

    // 满池周转: 占比低+序列尾部不涨 → 常驻(turnover判据)
    const uint64_t poolId = 0x65;
    std::vector<StackSeries> series5;
    series5.push_back(StepSeries(poolId, 1000, 1000, 5, beats));  // β_recent≈0
    std::unordered_map<uint64_t, StackCloseStats> stats5;
    stats5.emplace(poolId, MakeStats(10000, 100000000, 1000, 1000000, kLateTs));  // 占比0.01≤0.3
    const PerStackResult* p = ComputeOne(series5, stats5);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->status, CONF_STATUS_RESIDENT);
    EXPECT_EQ(p->residentCriterion, CONF_RESIDENT_TURNOVER);

    // 周转混入泄漏: 占比低(满足(4))但尾部上行(β_recent>0.05) → 非常驻+turnover标记
    const uint64_t mixedId = 0x66;
    StackSeries mixed;
    mixed.stackId = mixedId;
    for (int b = 0; b < beats; ++b)
    {
        int64_t v = 1000;
        if (b > 226)
        {
            v += 10 * static_cast<int64_t>(b - 227);  // 尾部每拍+10
        }
        AddRow(mixed, static_cast<uint32_t>(b), static_cast<uint64_t>(v));
    }
    std::vector<StackSeries> series6;
    series6.push_back(mixed);
    std::unordered_map<uint64_t, StackCloseStats> stats6;
    stats6.emplace(mixedId, MakeStats(10000, 100000000, 1000, 1000000, kLateTs));
    const PerStackResult* m = ComputeOne(series6, stats6);
    ASSERT_NE(m, nullptr);
    EXPECT_NE(m->status, CONF_STATUS_RESIDENT);
    EXPECT_TRUE(m->turnoverMark);

    // 占比高(慢泄漏): U/A=0.9>0.3 → 条件(4)不满足,非常驻(且无turnover标记)
    const uint64_t highRatioId = 0x67;
    std::unordered_map<uint64_t, StackCloseStats> stats7;
    stats7.emplace(highRatioId, MakeStats(10000, 100000000, 9000, 90000000, kLateTs));
    const PerStackResult* h = ComputeOne({}, stats7);
    ASSERT_NE(h, nullptr);
    EXPECT_NE(h->status, CONF_STATUS_RESIDENT);
    EXPECT_FALSE(h->turnoverMark);
}

// ---- UT-C7: 序列时间轴与乱序容错 ----
TEST(HostConfidenceTest, C7BeatTimeMapping)
{
    const uint64_t seriesStart = 1234567890;
    EXPECT_EQ(HostConfidence::BeatToTime(0, seriesStart, kBeatIntervalNs), seriesStart);  // 拍0对齐
    EXPECT_EQ(HostConfidence::BeatToTime(5, seriesStart, kBeatIntervalNs), seriesStart + 5 * kBeatIntervalNs);

    // 乱序输入容错: 拍序颠倒的行与升序输入产出相同结果
    const int beats = 301;
    const uint64_t stackId = 0x71;
    std::vector<StackSeries> sorted;
    sorted.push_back(LinearSeries(stackId, 1000, 1000, beats));
    std::vector<StackSeries> shuffled;
    StackSeries s;
    s.stackId = stackId;
    for (int b = beats - 1; b >= 0; --b)  // 逆序注入
    {
        AddRow(s, static_cast<uint32_t>(b), static_cast<uint64_t>(1000 + 1000 * b));
    }
    shuffled.push_back(s);
    std::unordered_map<uint64_t, StackCloseStats> stats;
    stats.emplace(stackId, MakeStats(beats, 1000ULL * beats, beats, 1000ULL * beats));
    const std::vector<PerStackResult> out1 =
        HostConfidence().Compute(sorted, stats, kWindowStart, kWindowEnd, false);
    const std::vector<PerStackResult> out2 =
        HostConfidence().Compute(shuffled, stats, kWindowStart, kWindowEnd, false);
    ASSERT_EQ(out1.size(), 1u);
    ASSERT_EQ(out2.size(), 1u);
    EXPECT_NEAR(out1[0].g, out2[0].g, kTol);
    EXPECT_NEAR(out1[0].lsi, out2[0].lsi, kTol);

    // 同拍重复行(后行胜): 重复行与单行序列结果一致
    std::vector<StackSeries> dup;
    StackSeries d;
    d.stackId = stackId;
    for (int b = 0; b < beats; ++b)
    {
        AddRow(d, static_cast<uint32_t>(b), static_cast<uint64_t>(1000 + 1000 * b));
        if (b % 3 == 0)
        {
            AddRow(d, static_cast<uint32_t>(b), static_cast<uint64_t>(1000 + 1000 * b));  // 同拍重发
        }
    }
    dup.push_back(d);
    const std::vector<PerStackResult> out3 =
        HostConfidence().Compute(dup, stats, kWindowStart, kWindowEnd, false);
    ASSERT_EQ(out3.size(), 1u);
    EXPECT_NEAR(out3[0].g, out1[0].g, kTol);
}

// ---- UT-C8: 降级标注 ----
TEST(HostConfidenceTest, C8Degraded)
{
    const int beats = 301;
    const uint64_t statsId = 0x81;
    const uint64_t freedLifeNs = 8 * 10 * 1000000000ULL;
    const uint64_t liveAgeNs = 10 * 100 * 1000000000ULL;
    auto baseStats = [&]()
    {
        std::unordered_map<uint64_t, StackCloseStats> m;
        m.emplace(statsId, MakeStats(18, 1800, 10, 1000, 0, freedLifeNs, liveAgeNs));
        return m;
    };

    // 无序列: NO_SERIES, G/P=0, R/S/E不受影响
    const PerStackResult* n = ComputeOne({}, baseStats());
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(n->degraded, CONF_DEGRADED_NO_SERIES);
    EXPECT_EQ(n->g, 0.0);
    EXPECT_EQ(n->p, 0.0);
    EXPECT_FALSE(n->growthValid);
    EXPECT_NEAR(n->s, 1.0, kTol);  // S仍按统计计算
    EXPECT_NEAR(n->r, 1000.0 / 1800.0, kTol);

    // 序列过短: INSUFFICIENT_SERIES
    std::vector<StackSeries> shortSeries;
    StackSeries ss;
    ss.stackId = statsId;
    AddRow(ss, 5, 1000);
    AddRow(ss, 6, 2000);
    shortSeries.push_back(ss);
    const PerStackResult* q = ComputeOne(shortSeries, baseStats());
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(q->degraded, CONF_DEGRADED_INSUFFICIENT_SERIES);
    EXPECT_EQ(q->g, 0.0);
    EXPECT_EQ(q->p, 0.0);

    // 槽被驱逐(序列完整但标注): SERIES_EVICTED, G/P=0中性,LSI由R/S/E支撑
    std::vector<StackSeries> evictedSeries;
    StackSeries ev;
    ev.stackId = statsId;
    ev.evicted = true;
    for (int b = 0; b < beats; ++b)
    {
        AddRow(ev, static_cast<uint32_t>(b), static_cast<uint64_t>(1000 + 1000 * b));
    }
    evictedSeries.push_back(ev);
    const PerStackResult* e = ComputeOne(evictedSeries, baseStats());
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(e->degraded, CONF_DEGRADED_SERIES_EVICTED);
    EXPECT_EQ(e->g, 0.0);
    EXPECT_EQ(e->p, 0.0);
    EXPECT_FALSE(e->growthValid);

    // 全窗无预热线程: NO_WARMUP_THREAD(统一降级,无视单栈序列)
    const PerStackResult* w = ComputeOne(evictedSeries, baseStats(), kWindowStart, kWindowEnd, true);
    ASSERT_NE(w, nullptr);
    EXPECT_EQ(w->degraded, CONF_DEGRADED_NO_WARMUP_THREAD);
}

// ---- UT-C10: 规模证据因子E与锚点验收 ----
TEST(HostConfidenceTest, C10ScaleFactorAndAnchors)
{
    // E_count边界: 1块→log10(2)/2≈0.15; 9块→0.5; 99块→1.0(clip)
    auto scale = [](uint64_t count, uint64_t bytes)
    {
        const uint64_t stackId = 0x101;
        std::unordered_map<uint64_t, StackCloseStats> m;
        m.emplace(stackId, MakeStats(count, bytes, count, bytes));
        const std::vector<PerStackResult> out =
            HostConfidence().Compute({}, m, kWindowStart, kWindowEnd, false);
        return out.empty() ? -1.0 : out[0].e;
    };
    EXPECT_NEAR(scale(1, 1), std::log10(2.0) / 2.0, kTol);          // 字节维度≈0
    EXPECT_NEAR(scale(9, 9), 0.5, 1e-9);                             // log10(10)/2
    EXPECT_NEAR(scale(99, 99), 1.0, 1e-9);                           // log10(100)/2=1
    // E_bytes边界: 50MiB→log10(51)/2.8≈0.61; 500MiB→≈0.96; 1GiB→clip 1.0
    EXPECT_NEAR(scale(1, 50 * 1048576ULL), std::log10(51.0) / 2.8, 1e-9);
    EXPECT_NEAR(scale(1, 500 * 1048576ULL), std::log10(501.0) / 2.8, 1e-9);
    EXPECT_NEAR(scale(1, 1073741824ULL), 1.0, 1e-9);
    // 双维度取大: 100块×100B → 次数维度胜
    EXPECT_NEAR(scale(100, 100), 1.0, 1e-9);
    // 次数递增→E单调递增(至99次饱和)
    double prev = -1.0;
    for (uint64_t c = 1; c <= 98; ++c)
    {
        const double e = scale(c, 1048576ULL * c);  // 每块1MiB
        EXPECT_GE(e, prev);
        prev = e;
    }

    // 锚点A1: 20次申请不释放(1MiB/次,早期突发) → LSI=27+29×0.661≈46>20,且判常驻
    const uint64_t a1Id = 0x102;
    std::vector<StackSeries> series;
    series.push_back(StepSeries(a1Id, 0, 20 * 1048576ULL, 5, 301));  // 早期突发后平稳,G≈0
    std::unordered_map<uint64_t, StackCloseStats> stats;
    stats.emplace(a1Id, MakeStats(20, 20 * 1048576ULL, 20, 20 * 1048576ULL, 10 * kBeatIntervalNs));
    const PerStackResult* a1 = ComputeOne(series, stats);
    ASSERT_NE(a1, nullptr);
    EXPECT_NEAR(a1->lsi, 27.0 + 29.0 * std::log10(21.0) / 2.0, 0.6);
    EXPECT_GT(a1->lsi, 20.0);
    EXPECT_EQ(a1->status, CONF_STATUS_RESIDENT);  // 早期分配+E高仍常驻(E不参与判定)

    // 锚点A2: 100次不释放累计100MiB(>50MiB) → LSI=56>50
    const uint64_t a2Id = 0x103;
    std::vector<StackSeries> series2;
    series2.push_back(StepSeries(a2Id, 0, 100 * 1048576ULL, 5, 301));
    std::unordered_map<uint64_t, StackCloseStats> stats2;
    stats2.emplace(a2Id, MakeStats(100, 100 * 1048576ULL, 100, 100 * 1048576ULL, 10 * kBeatIntervalNs));
    const PerStackResult* a2 = ComputeOne(series2, stats2);
    ASSERT_NE(a2, nullptr);
    EXPECT_GT(a2->lsi, 50.0);

    // 锚点A3: 20次不释放累计500MiB → E_bytes=0.96 → LSI≈55>50
    const uint64_t a3Id = 0x104;
    std::vector<StackSeries> series3;
    series3.push_back(StepSeries(a3Id, 0, 500 * 1048576ULL, 5, 301));
    std::unordered_map<uint64_t, StackCloseStats> stats3;
    stats3.emplace(a3Id, MakeStats(20, 500 * 1048576ULL, 20, 500 * 1048576ULL, 10 * kBeatIntervalNs));
    const PerStackResult* a3 = ComputeOne(series3, stats3);
    ASSERT_NE(a3, nullptr);
    EXPECT_GT(a3->lsi, 50.0);

    // 锚点A4: 次数1→98递增(每块1MiB,不释放) → LSI单调递增
    double prevLsi = -1.0;
    for (uint64_t c = 1; c <= 98; ++c)
    {
        const uint64_t sid = 0x105;
        std::unordered_map<uint64_t, StackCloseStats> m;
        m.emplace(sid, MakeStats(c, c * 1048576ULL, c, c * 1048576ULL, 10 * kBeatIntervalNs));
        const std::vector<PerStackResult> out =
            HostConfidence().Compute({}, m, kWindowStart, kWindowEnd, false);
        ASSERT_EQ(out.size(), 1u);
        EXPECT_GE(out[0].lsi, prevLsi);
        prevLsi = out[0].lsi;
    }
}
