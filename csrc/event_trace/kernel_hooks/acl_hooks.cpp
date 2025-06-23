// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "acl_hooks.h"
#include <cstdint>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <unordered_map>

#include "cpython.h"
#include "event_report.h"
#include "vallina_symbol.h"
#include "serializer.h"
#include "log.h"
#include "record_info.h"
#include "bit_field.h"

using namespace Leaks;

static void LeaksPythonCall(const std::string& module, const std::string& function)
{
    if (!Utility::IsPyInterpRepeInited()) {
            CLIENT_ERROR_LOG("Python Interpreter initialization FAILED");
            return;
    }
 
    Utility::PyInterpGuard stat;
    Utility::PythonObject sys = Utility::PythonObject::Import("sys");
    Utility::PythonObject modules = sys.Get("modules");
    Utility::PythonObject torch = modules.GetItem(Utility::PythonObject("torch"));
    if (!torch.IsBad()) {
        Utility::PythonObject pythonModule = Utility::PythonObject::Import(module, false);
        if (pythonModule.IsBad()) {
            CLIENT_ERROR_LOG("import " + module + " FAILED");
            return;
        }
 
        Utility::PythonObject pythonFunction = pythonModule.Get(function);
        if (pythonFunction.IsBad()) {
            CLIENT_ERROR_LOG("cannot get function " + function);
            return;
        }
        pythonFunction.Call();
    }
    return;
}
 
ACL_FUNC_VISIBILITY aclError aclInit(const char *configPath)
{
    using AclInit = decltype(&aclInit);
    auto vallina = VallinaSymbol<AclLibLoader>::Instance().Get<AclInit>("aclInit");
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return ACL_ERROR_INTERNAL_ERROR;
    }

    aclError ret = vallina(configPath);
    if (!EventReport::Instance(CommType::SOCKET).ReportAclItf(AclOpType::INIT)) {
        CLIENT_ERROR_LOG("aclInit report FAILED");
    }

    // 命令行判断是否为OP级别
    Config userConfig =  EventReport::Instance(CommType::SOCKET).GetConfig();
    BitField<decltype(userConfig.levelType)> levelType(userConfig.levelType);
    if (levelType.checkBit(static_cast<size_t>(LevelType::LEVEL_OP))) {
        LeaksPythonCall("msleaks.aten_collection", "enable_aten_collector");
    }

    BitField<decltype(userConfig.analysisType)> analysisType(userConfig.analysisType);
    if (analysisType.checkBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        LeaksPythonCall("msleaks.optimizer_step_hook", "enable_optimizer_step_hook");
    }

    return ret;
}

ACL_FUNC_VISIBILITY aclError aclFinalize()
{
    using AclFinalize = decltype(&aclFinalize);
    auto vallina = VallinaSymbol<AclLibLoader>::Instance().Get<AclFinalize>("aclFinalize");
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return ACL_ERROR_INTERNAL_ERROR;
    }

    aclError ret = vallina();
    if (!EventReport::Instance(CommType::SOCKET).ReportAclItf(AclOpType::FINALIZE)) {
        CLIENT_ERROR_LOG("aclInit report FAILED");
    }

    Config userConfig =  EventReport::Instance(CommType::SOCKET).GetConfig();
    BitField<decltype(userConfig.analysisType)> analysisType(userConfig.analysisType);
    if (analysisType.checkBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        LeaksPythonCall("msleaks.optimizer_step_hook", "disable_optimizer_step_hook");
    }

    return ret;
}
