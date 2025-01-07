// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "command.h"
#include <map>
#include "process.h"
#include "protocol.h"

namespace Leaks {

void RecordHandler(const ClientId &clientId, const EventRecord &record, AnalyzerFactory &analyzerfactory)
{
    switch (record.type) {
        case RecordType::MSTX_MARK_RECORD:
            MstxAnalyzer::Instance().RecordMstx(clientId, record.record.mstxRecord);
            break;
        case RecordType::KERNEL_LAUNCH_RECORD:
            Utility::LogInfo(
                "kernelLaunch record, name: %s, index: %u, type: %u, time: %u, streamId: %d, blockDim: %u",
                record.record.kernelLaunchRecord.kernelName,
                record.record.kernelLaunchRecord.kernelLaunchIndex,
                record.record.kernelLaunchRecord.type,
                record.record.kernelLaunchRecord.timeStamp,
                record.record.kernelLaunchRecord.streamId,
                record.record.kernelLaunchRecord.blockDim);
            break;
        case RecordType::ACL_ITF_RECORD:
            Utility::LogInfo("aclItf record, index: %u, type: %u, time: %u",
                record.record.aclItfRecord.aclItfRecordIndex,
                record.record.aclItfRecord.type,
                record.record.aclItfRecord.timeStamp);
            break;
        default:
            auto analyzer = analyzerfactory.CreateAnalyzer(record.type);
            analyzer->Record(clientId, record);
            break;
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