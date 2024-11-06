// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "command.h"
#include "process.h"
#include "protocol.h"
#include "analysis/analyzer.h"
namespace Leaks {

void Command::Exec(const std::vector<std::string> &execParams) const
{
    Process process;
    Protocol protocol {};
    Analyzer analyzer(config_);

    auto analysisFuc = [&protocol, &analyzer](const std::string &manyMsg) {
        protocol.Feed(manyMsg);
        auto packet = protocol.GetPacket();
        analyzer.Do(packet.GetPacketBody());

        return;
    };

    process.RegisterAnalysisFuc(analysisFuc);
    process.Launch(execParams);

    return;
}

}