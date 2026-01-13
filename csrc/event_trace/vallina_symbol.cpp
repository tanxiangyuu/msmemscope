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

#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <vector>
#include <cstring>
#include "vallina_symbol.h"

namespace MemScope {
    void *LibLoad(std::string libName)
    {
        if (libName.empty()) {
            std::cout << "[msmemscope] Error: Null library name." << std::endl;
            return nullptr;
        }
        std::string libPath = libName;
        const char *pathEnv = std::getenv("ASCEND_HOME_PATH");
        if (pathEnv && !std::string(pathEnv).empty()) {
            libPath = pathEnv;
            libPath += "/lib64/" + libName;
            return dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
        }
        // 找不到Ascend Path
        std::cout << "[msmemscope] Error: Failed to acquire ASCEND_HOME_PATH environment variable while loading "
            << libName << ". Try to load lib directly."
            << std::endl;
        return dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    }
    
    void *GetSymbol(char const *symbol)
    {
        void *func = dlsym(nullptr, symbol);
        return func;
    }
}
