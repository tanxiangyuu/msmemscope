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

#ifndef VALLINA_SYMBOL_H
#define VALLINA_SYMBOL_H

#include <string>
#include <array>
#include <iostream>
#include <dlfcn.h>

namespace MemScope {

void *LibLoad(std::string libName);

void *GetSymbol(char const *symbol);

/* VallinaSymbol 类用于从指定的动态库句柄中获取函数符号
 * @tparam LibLoader 动态库加载器，需要实现 Load 方法
 */
template <typename LibLoader>
class VallinaSymbol {
public:
    inline static VallinaSymbol &Instance(void)
    {
        static VallinaSymbol ins;
        return ins;
    }
    VallinaSymbol(VallinaSymbol const &) = delete;
    VallinaSymbol &operator=(VallinaSymbol const &) = delete;

    /* 获取指定函数名的符号地址
     * @param symbol 要获取的函数符号名
     * @return 获取到的函数符号
     */
    inline void *Get(char const *symbol) const
    {
        if (handle_ == nullptr) {
            std::cout << "[msmemscope] lib handle is NULL" << std::endl;
            return nullptr;
        }
        return dlsym(handle_, symbol);
    }

    /* 获取指定函数名的符号地址，并且转换为对应类型的函数指针
     * @param symbol 要获取的函数符号名
     * @return 获取到的函数符号
     */
    template <typename Func>
    inline Func Get(char const *symbol) const
    {
        return reinterpret_cast<Func>(Get(symbol));
    }

private:
    inline VallinaSymbol(void) : handle_(LibLoader::Load())
    {}

private:
    void *handle_;
};
}
#endif
