// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef LEAKS_ATB_ENUM_TO_STRING_H
#define LEAKS_ATB_ENUM_TO_STRING_H

#include <string>

#include "atb_stub.h"
#include "mki_stub.h"

namespace atb {
const std::string& MemScopeEnumToString(aclDataType value);
const std::string& MemScopeEnumToString(aclFormat value);
const std::string& MemScopeEnumToString(Mki::TensorDType value);
const std::string& MemScopeEnumToString(Mki::TensorFormat value);
}

#endif