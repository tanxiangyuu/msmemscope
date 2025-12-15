/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

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