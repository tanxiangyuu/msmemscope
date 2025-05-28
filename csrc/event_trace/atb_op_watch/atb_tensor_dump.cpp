// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include <fstream>
#include <vector>
#include "utils.h"
#include "kernel_hooks/acl_hooks.h"
#include "file.h"
#include "event_report.h"
#include "atb_tensor_dump.h"
#include "calculate_md5.h"

namespace Leaks {
void CleanFileName(std::string& fileName)
{
    for (size_t i = 0; i < fileName.size(); i++) {
        if (fileName[i] == '/') {
            fileName[i] = '.';
        }
    }
    return;
}

ATBTensorDump::ATBTensorDump()
{
    Config config = EventReport::Instance(CommType::SOCKET).GetConfig();
    fullContent_ = config.watchConfig.fullContent;

    dumpDir_ = std::string(config.outputDir) + "/atb_op_dump";
    if (!Utility::MakeDir(dumpDir_)) {
        CLIENT_ERROR_LOG("Make dir failed.");
    }

    // 当只落盘哈希值时，在构造时初始化csv文件
    if (!IsDumpFullContent()) {
        int32_t devId = GD_INVALID_NUM;
        if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
            CLIENT_ERROR_LOG("Get device id failed, " + std::to_string(devId));
        }
        std::string fileName = "watch_dump_md5_" + std::to_string(devId) + "_";
        std::string tableHeader = "tensor info,MD5\n";
        if (!Utility::CreateCsvFile(&csvFile_, dumpDir_, fileName, tableHeader)) {
            CLIENT_ERROR_LOG("Create csv file failed.");
        }
    }
}

ATBTensorDump::~ATBTensorDump()
{
    if (csvFile_ != nullptr) {
        fclose(csvFile_);
        csvFile_ = nullptr;
    }
}

bool ATBTensorDump::IsDumpFullContent()
{
    return fullContent_;
}

bool ATBTensorDump::DumpTensorBinary(const std::vector<char> &hostData, std::string& fileName)
{
    CleanFileName(fileName);

    Utility::UmaskGuard guard{Utility::DEFAULT_UMASK_FOR_BIN_FILE};
    std::ofstream outFile(dumpDir_ + "/" + fileName, std::ios::binary);
    if (!outFile) {
        return false;
    }

    outFile.write(hostData.data(), hostData.size());

    if (!outFile.good()) {
        return false;
    }

    outFile.close();

    return true;
}

bool ATBTensorDump::DumpTensorMD5(const std::vector<char> &hostData, std::string& fileName)
{
    auto MD5Value = Utility::GetTensorMD5(hostData);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!Utility::Fprintf(csvFile_, "%s,%s\n", fileName.c_str(), MD5Value.c_str())) {
        CLIENT_ERROR_LOG("Write tensor md5 info failed.");
        return false;
    }
    return true;
}

bool ATBTensorDump::Dump(const Tensor& tensor, std::string& fileName)
{
    using AclrtMemcpy = decltype(&aclrtMemcpy);
    auto vallina = VallinaSymbol<AclLibLoader>::Instance().Get<AclrtMemcpy>("aclrtMemcpy");
    if (vallina == nullptr) {
        return false;
    }

    std::vector<char> hostData(tensor.dataSize);
    aclError ret = vallina(hostData.data(), tensor.dataSize, tensor.data, tensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return false;
    }

    if (IsDumpFullContent()) {
        return DumpTensorBinary(hostData, fileName);
    }

    return DumpTensorMD5(hostData, fileName);
}

}