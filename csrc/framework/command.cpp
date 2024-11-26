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
        if (manyMsg == PROCESS_EXIT_MSG) {
            analyzer.LeakAnalyze();
            return;
        }
        protocol.Feed(manyMsg);
        while (true) {
            auto packet = protocol.GetPacket();
            switch (packet.GetPacketHead().type) {
                case PacketType::RECORD:
                    analyzer.Do(packet.GetPacketBody());
                    break;
                case PacketType::INVALID:
                default:
                    return;
            }
        }
        analyzer.LeakAnalyze();
        
        return;
    };

    process.RegisterAnalysisFuc(analysisFuc);
    process.StartListen();
    process.Launch(execParams);

    return;
}

}