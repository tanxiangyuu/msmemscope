// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "command.h"
#include <map>
#include "process.h"
#include "protocol.h"
#include "utils.h"

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

void StepInterCompare(const std::vector<std::string> &paths)
{
    StepInterAnalyzer stepInterAnalyzer;
    Utility::LogInfo("Start to analyze stepinter memory data, please wait!");
    auto start_time = Utility::GetTimeMicroseconds();
    stepInterAnalyzer.StepInterOfflineCompare(paths);
    auto end_time = Utility::GetTimeMicroseconds();
    Utility::LogInfo("The stepinter memory analysis has been completed"
        "in a total time of %.6f(s)", (end_time-start_time) / MICROSEC);
    return ;
}

void Command::Exec() const
{
    if (userCommand_.config.enableCompare) {
        StepInterCompare(userCommand_.paths);
        return;
    }
    
    Process process(userCommand_.config);
    std::map<ClientId, Protocol> protocolList;
    AnalyzerFactory analyzerfactory{userCommand_.config};

    auto msgHandler = [&protocolList, &analyzerfactory](ClientId &clientId,
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
                    DumpRecord::GetInstance().DumpData(clientId, packet.GetPacketBody().eventRecord);
                    TraceRecord::GetInstance().TraceHandler(packet.GetPacketBody().eventRecord);
                    RecordHandler(clientId, packet.GetPacketBody().eventRecord, analyzerfactory);
                    break;
                case PacketType::LOG: {
                    auto log = packet.GetPacketBody().log;
                    Utility::LogRecv("%s", std::string(log.buf, log.buf + log.len).c_str());
                }
                case PacketType::INVALID:
                default:
                    return;
            }
        }

        return;
    };
    process.RegisterMsgHandlerHook(msgHandler);
    process.Launch(userCommand_.cmd);

    return;
}

}