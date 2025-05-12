// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include <fstream>
#include <vector>
#include "utils.h"
#include "kernel_hooks/acl_hooks.h"
#include "file.h"
#include "atb_tensor_dump.h"

namespace Leaks {
static void CleanFileName(std::string& fileName)
{
    for (size_t i = 0; i < fileName.size(); i++) {
        if (fileName[i] == '/') {
            fileName[i] = '.';
        }
    }
    return;
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

    // 后续ouput路径需要由server传入, 且完成目录安全校验、输出文件安全校验
    std::string dumpDir = "atb_op_dump";
    (void)Utility::MakeDir(dumpDir);
    CleanFileName(fileName);
    std::ofstream outFile(dumpDir + "/" + fileName, std::ios::binary);
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

}