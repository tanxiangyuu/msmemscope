// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef LEAKS_ATB_ENUM_TO_STRING_H
#define LEAKS_ATB_ENUM_TO_STRING_H

#include <string>

#include "atb_stub.h"
#include "mki_stub.h"

namespace atb {
const std::string& LeaksEnumToString(aclDataType value);
const std::string& LeaksEnumToString(aclFormat value);
const std::string& LeaksEnumToString(Mki::TensorDType value);
const std::string& LeaksEnumToString(Mki::TensorFormat value);
}

#endif