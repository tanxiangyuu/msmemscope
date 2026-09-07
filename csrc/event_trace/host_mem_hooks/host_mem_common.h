/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#ifndef MSMEMSCOPE_HOST_MEM_COMMON_H
#define MSMEMSCOPE_HOST_MEM_COMMON_H

/*
 * host内存钩子so内共享类型与真分配桥
 *
 * host_mem_hooks.cpp的实现整体处于匿名命名空间(内部链接,防与宿主符号冲突),
 * 成员不可跨TU复用;py采集模块(py_stack_capture)需要:
 *   1. 栈表条目定义(StackRecord)——采集钩点/闭窗组装/清理均需操作条目字段;
 *   2. 真函数分配底座(RealMallocAllocator)——py串缓冲与缓存容器必须不经PLT
 *      (场景B硬性要求,见host_mem_hooks.cpp文件头注释)。
 * 分配实现保留在host_mem_hooks.cpp(竞技场兜底/抑制守卫/惰性解析单一实现),
 * 经桥函数SharedRealAlloc/SharedRealFree转调,桥实现见host_mem_hooks.cpp文件尾。
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>

namespace hostmem
{

// py采集三态: 热路径仅NONE/PYTHON两态迁移,NA为闭窗派生——泄漏点无py采集时
// 按闭窗精确unfreed判据派生的诚实标注
enum PyStackState : uint32_t
{
    PY_STACK_NONE = 0,      // 未采集(不满足约束/未达标/无GIL待重试/解释器不可用)
    PY_STACK_CAPTURED = 1,  // 已采集(PYTHON热路径)
    PY_STACK_NA = 2         // 闭窗派生(仅闭窗冻结态使用,热路径不写)
};

// 栈表条目(原host_mem_hooks.cpp匿名命名空间StackEntry提升为共享,新增py字段):
// py采集模块与栈表主文件双侧可见,避免跨文件重复定义与不透明指针传参。
// 字段契约见host_mem_hooks.cpp文件头引用契约注释:
struct StackRecord
{
    uint64_t stackId = 0;  // 单调递增,低6位=所属分片号,0保留未知栈
    // 全深度PC(登记慢路径采集,闭窗聚合符号化用;符号化后释放)与帧数
    uintptr_t* fullPcs = nullptr;
    uint32_t fullCount = 0;
    bool warmedUp = false;  // 稳态预热已选中过(分片锁内写/读)。不能用缓存命中
                            // 判"已预热":fullPcs[0]是hook锚点帧,全栈共享同一
                            // pc,首个栈预热后其余栈全被误判跳过
    // 引用计数refs(在表pin+存活块数+在途lookup数): 1=仅pin,零存活块零在途;
    // refs==1条目可被淘汰(栈表满→死栈回收,见EvictDeadStackLocked)。增减relaxed:
    // 所有+1在栈分片锁临界区内(除InsertBlock块引用+1在块表锁内,有调用方在途
    // ref兜底,期间refs恒>=2);-1恰一次于dispose(块释放/捕获后终态)。块表持
    // owner指针期间refs>=2(块引用兜底),与淘汰判读(refs==1)互斥
    std::atomic<uint32_t> refs{1};        // 1=在表pin + 存活块数 + 在途lookup数
    std::atomic<int64_t> liveBytes{0};    // 当前存活字节(泄漏量真相源,与闭窗
                                          // 报告unfreedBytes同构——top泄漏点
                                          // 符号采样的排序键)
    std::atomic<uint64_t> allocCount{0};  // 本窗口内申请次数(InsertBlock临界区内自增)
    std::atomic<uint64_t> allocBytes{0};  // 本窗口内申请字节(同上)
    // py调用栈: pyBuf为RealMalloc底座串。写者唯一性: 采集在持GIL线程上执行,
    // GIL互斥两个采集线程;发布=release store pyState(CAPTURED),
    // store前的pyBuf写入对acquire读者可见。读者(闭窗组装/淘汰/清表)仅在
    // pyState==CAPTURED时读pyBuf/释放pyBuf,与写者无竞态(见CaptureIfEnabled注释)
    char* pyBuf = nullptr;
    uint32_t pyLen = 0;
    std::atomic<uint32_t> pyState{PY_STACK_NONE};
};

// 真分配桥(场景B硬性要求,语义同host_mem_hooks.cpp RealMalloc族): 钩子so内
// 对malloc的直接调用经PLT解析回本so钩子函数(先加载者胜),必须经dlsym
// (RTLD_NEXT)取真函数。实现保留在host_mem_hooks.cpp(匿名命名空间成员同TU
// 可见,竞技场兜底/抑制守卫单一实现),py_stack_capture.cpp经此桥分配py串
// 缓冲与缓存容器,绝不经PLT
void* SharedRealAlloc(size_t size);
void SharedRealFree(void* ptr);
size_t SharedRealUsableSize(void* ptr);

// 钩子内自定义allocator(场景B): 分配经SharedRealAlloc不经PLT,构造早期由
// 竞技场兜底(SharedRealAlloc内部);失败抛bad_alloc交调用方降级(丢包计数)——绝不trap杀宿主:
// 钩子寄生在任意宿主进程,崩溃请求可能源自内部异常状态(如哈希表负桶数请求),
// 宿主自身的malloc可能完全正常。原host_mem_hooks.cpp匿名命名空间定义,现提升
// 为共享(py缓存容器同一底座)
template <typename T>
struct RealMallocAllocator
{
    using value_type = T;

    RealMallocAllocator() = default;
    template <typename U>
    RealMallocAllocator(const RealMallocAllocator<U>&)
    {
    }

    T* allocate(std::size_t n)
    {
        void* p = SharedRealAlloc(n * sizeof(T));
        if (p == nullptr)
        {
            throw std::bad_alloc();  // 真OOM或请求溢出:由调用方捕获降级
        }
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t) { SharedRealFree(p); }
};

template <typename T, typename U>
bool operator==(const RealMallocAllocator<T>&, const RealMallocAllocator<U>&)
{
    return true;
}

template <typename T, typename U>
bool operator!=(const RealMallocAllocator<T>&, const RealMallocAllocator<U>&)
{
    return false;
}

}  // namespace hostmem

#endif /* MSMEMSCOPE_HOST_MEM_COMMON_H */
