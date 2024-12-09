// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef __HOOK_GLOBAL_HANDLE_H__
#define __HOOK_GLOBAL_HANDLE_H__

namespace Leaks {

enum class KernelType : uint8_t {
    AICPU = 0,        /// 对应 RTS 中的 RT_DEV_BINARY_MAGIC_ELF_AICPU
    AIVEC,            /// 对应 RTS 中的 RT_DEV_BINARY_MAGIC_ELF_AIVEC
    AICUBE,           /// 对应 RTS 中的 RT_DEV_BINARY_MAGIC_ELF_AICUBE
    MIX,              /// 对应 RTS 中的 RT_DEV_BINARY_MAGIC_ELF
};

struct BinKernel {
    std::vector<char> bin;
    KernelType kernelType;
};

// 使用单例而不是全局变量维护两个映射表，避免neo场景的segmentation fault，后续劫持接口处的所有全局变量统一用该单例例维护
class GlobalHandle {
public:
    static GlobalHandle &GetInstance()
    {
        static GlobalHandle instance;
        return instance;
    }

    ~GlobalHandle()
    {
        /// 预期在UnRegister函数中，通过erase删去映射关系
        /// 某些情况下，单例在UnRegister函数调用前被析构，析构后仍然可以访问，属于非法访问内存
        /// 为了避免非法访问时erase造成segmentation fault，在析构时clear，保证析构后无法访问到handle
        handleBinKernelMap_.clear();
        stubHandleMap_.clear();
    }

    /* 两个映射表：
    * 1.HandleBinKernelMap 用于绑定算子 handle 与 kernel 二进制以及 KernelType 的映射关系
    * 2.StubHandleMap 用于绑定算子 stubFunc 与 handle 的映射关系
    */
    using HandleBinKernelMapType = std::map<const void*, BinKernel>;
    using StubHandleMapType = std::map<const void*, const void*>;

    HandleBinKernelMapType handleBinKernelMap_;
    StubHandleMapType stubHandleMap_;

private:
    GlobalHandle() = default;
    GlobalHandle(const GlobalHandle&) = delete;
    GlobalHandle& operator=(const GlobalHandle&) = delete;
};

}
#endif
