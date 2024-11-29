// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "command.h"
#include <map>
#include "process.h"
#include "protocol.h"
#include "analysis/analyzer.h"

namespace Leaks {

void Command::Exec(const std::vector<std::string> &execParams) const
{
    Process process;
    std::map<ClientId, Protocol> protocolList;
    Analyzer analyzer(config_);

    auto msgHandler = [&protocolList, &analyzer](ClientId &clientId, std::string &manyMsg) {
        if (protocolList.find(clientId) == protocolList.end()) {
            protocolList.insert({clientId, Protocol{}});
        }
        Protocol protocol = protocolList[clientId];
        protocol.Feed(manyMsg);
        while (true) {
            auto packet = protocol.GetPacket();
            switch (packet.GetPacketHead().type) {
                case PacketType::RECORD:
                    analyzer.Do(clientId, packet.GetPacketBody());
                    break;
                case PacketType::INVALID:
                default:
                    return;
            }
        }
        
        return;
    };

    process.RegisterMsgHandlerHook(msgHandler);
    process.Launch(execParams);
    return;
}

}