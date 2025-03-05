// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_VALLINA_SYMBOL_H
#define LEAKS_VALLINA_SYMBOL_H

#include <string>
#include <iostream>
#include <dlfcn.h>

namespace Leaks {

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
            std::cout << "[msleaks] lib handle is NULL" << std::endl;
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
