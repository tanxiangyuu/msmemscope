// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef FRAMEWORK_PROTOCOL_H
#define FRAMEWORK_PROTOCOL_H

#include <string>
#include "config_info.h"
#include "record_info.h"

namespace Leaks {
// 承载解包后的信息，用于传递到分析模块进行处理
struct Packet {
    EventRecord record;
};

// Protocol类接收数据，根据协议解包，后期可根据分析算法的不同进行扩展
class Protocol {
public:
    explicit Protocol(const AnalysisConfig &config) : config_(config) {};
    ~Protocol() = default;
    void Feed(std::string const &msg);
    Packet GetPacket(void);
private:
    AnalysisConfig config_;
};

}

#endif