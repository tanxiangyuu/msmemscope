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

#ifndef FRAMEWORK_COMMAND_H
#define FRAMEWORK_COMMAND_H

#include <vector>
#include <string>
#include <memory>
#include "config_info.h"

namespace MemScope {

// Command类主要针对解析后的命令进行处理，是串接流程的主要类
class Command {
public:
    explicit Command(const UserCommand &userCommand) : userCommand_{userCommand} {}
    void Exec() const;
private:
    UserCommand userCommand_;
};

}

#endif
