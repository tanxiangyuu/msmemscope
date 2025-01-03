// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "command.h"
#include <map>
#include "process.h"
#include "protocol.h"

namespace Leaks {

void RecordHandler(const ClientId &clientId, const EventRecord &record, AnalyzerFactory &analyzerfactory)
{
    // mstx类记录
    if (record.type == RecordType::MSTX_MARK_RECORD) {
        auto mstxRecord = record.record.mstxRecord;
        MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecord);
        return;
    }

    // 分析类多态实现
    auto analyzer = analyzerfactory.CreateAnalyzer(record.type);
    if (analyzer) {
        analyzer->Record(clientId, record);
    } else {
        /* now acl or kernel */
    }

    return;
}

void DumpHandler(const ClientId &clientId, DumpRecord &dump, const EventRecord &record)
{
    if (!dump.DumpData(clientId, record)) {
        Utility::LogError("dump data fail");
    }
}

void Command::Exec(const std::vector<std::string> &execParams) const
{
    Process process(config_);
    std::map<ClientId, Protocol> protocolList;
    AnalyzerFactory analyzerfactory{config_};
    DumpRecord dump{};

    auto msgHandler = [&protocolList, &dump, &analyzerfactory](ClientId &clientId,
    std::string &manyMsg) {
        if (protocolList.find(clientId) == protocolList.end()) {
            protocolList.insert({clientId, Protocol{}});
        }
        Protocol protocol = protocolList[clientId];
        protocol.Feed(manyMsg);
        while (true) {
            auto packet = protocol.GetPacket();
            switch (packet.GetPacketHead().type) {
                case PacketType::RECORD:
                    DumpHandler(clientId, dump, packet.GetPacketBody());
                    TraceRecord::GetInstance().TraceHandler(packet.GetPacketBody());
                    RecordHandler(clientId, packet.GetPacketBody(), analyzerfactory);
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