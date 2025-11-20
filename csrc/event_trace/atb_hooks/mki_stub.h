// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#ifndef LEAKS_MKI_STUB_H
#define LEAKS_MKI_STUB_H

#include <vector>
#include <exception>
#include <stdexcept>

namespace Mki {
constexpr size_t DEFAULT_SVECTOR_SIZE = 48;
constexpr bool CHECK_BOUND = true;

struct MaxSizeExceeded : public std::exception {};
template <class T, std::size_t MAX_SIZE = DEFAULT_SVECTOR_SIZE> class SVector {
public:
    T *begin() noexcept
    {
        return &storage_[0];
    }

    const T *begin() const noexcept
    {
        return &storage_[0];
    }

    T *end() noexcept
    {
        return (&storage_[0]) + size_;
    }

    const T *end() const noexcept
    {
        return (&storage_[0]) + size_;
    }

    const T &operator[](std::size_t i) const
    {
        if (size_ == 0 || i >= size_) {
            throw std::out_of_range("out of range");
        }
        return storage_[i];
    }

    void push_back(const T &val) noexcept((!CHECK_BOUND) && std::is_nothrow_assignable<T, const T &>::value)
    {
        if (CHECK_BOUND && size_ == MAX_SIZE) {
            throw MaxSizeExceeded();
        }
        storage_[size_++] = val;
    }

    std::size_t size() const noexcept { return size_; }
private:
    T storage_[MAX_SIZE + 1];
    std::size_t size_{0};
};

enum TensorDType : int {
    TENSOR_DTYPE_UNDEFINED = -1,
    TENSOR_DTYPE_FLOAT = 0,
    TENSOR_DTYPE_FLOAT16 = 1,
    TENSOR_DTYPE_INT8 = 2,
    TENSOR_DTYPE_INT32 = 3,
    TENSOR_DTYPE_UINT8 = 4,
    TENSOR_DTYPE_INT16 = 6,
    TENSOR_DTYPE_UINT16 = 7,
    TENSOR_DTYPE_UINT32 = 8,
    TENSOR_DTYPE_INT64 = 9,
    TENSOR_DTYPE_UINT64 = 10,
    TENSOR_DTYPE_DOUBLE = 11,
    TENSOR_DTYPE_BOOL = 12,
    TENSOR_DTYPE_STRING = 13,
    TENSOR_DTYPE_COMPLEX64 = 16,
    TENSOR_DTYPE_COMPLEX128 = 17,
    TENSOR_DTYPE_BF16 = 27
};

enum TensorFormat : int {
    TENSOR_FORMAT_UNDEFINED = -1,
    TENSOR_FORMAT_NCHW = 0,
    TENSOR_FORMAT_NHWC = 1,
    TENSOR_FORMAT_ND = 2,
    TENSOR_FORMAT_NC1HWC0 = 3,
    TENSOR_FORMAT_FRACTAL_Z = 4,
    TENSOR_FORMAT_NC1HWC0_C04 = 12,
    TENSOR_FORMAT_HWCN = 16,
    TENSOR_FORMAT_NDHWC = 27,
    TENSOR_FORMAT_FRACTAL_NZ = 29,
    TENSOR_FORMAT_NCDHW = 30,
    TENSOR_FORMAT_NDC1HWC0 = 32,
    TENSOR_FORMAT_FRACTAL_Z_3D = 33
};

struct TensorDesc {
    TensorDType dtype = TENSOR_DTYPE_UNDEFINED;
    TensorFormat format = TENSOR_FORMAT_UNDEFINED;
    Mki::SVector<int64_t> dims;
    std::vector<int64_t> strides;
    int64_t offset = 0;
};

struct Tensor {
    TensorDesc desc;
    void *data = nullptr;
    size_t dataSize = 0;
    void *hostData = nullptr;
};
class LaunchParam {};
using MemScopeOriginalGetInTensors = const Mki::SVector<Mki::Tensor>& (*)(Mki::LaunchParam*);
using MemScopeOriginalGetOutTensors = const Mki::SVector<Mki::Tensor>& (*)(Mki::LaunchParam*);
} // namespace Mki

#endif