// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#ifndef LEAKS_ATB_STUB_H
#define LEAKS_ATB_STUB_H

#include <functional>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *aclrtStream;

typedef enum {
    ACL_DT_UNDEFINED = -1,
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_INT8 = 2,
    ACL_INT32 = 3,
    ACL_UINT8 = 4,
    ACL_INT16 = 6,
    ACL_UINT16 = 7,
    ACL_UINT32 = 8,
    ACL_INT64 = 9,
    ACL_UINT64 = 10,
    ACL_DOUBLE = 11,
    ACL_BOOL = 12,
    ACL_STRING = 13,
    ACL_COMPLEX64 = 16,
    ACL_COMPLEX128 = 17,
    ACL_BF16 = 27,
    ACL_INT4 = 29,
    ACL_UINT1 = 30,
    ACL_COMPLEX32 = 33,
    ACL_HIFLOAT8 = 34,
    ACL_FLOAT8_E5M2 = 35,
    ACL_FLOAT8_E4M3FN = 36,
    ACL_FLOAT8_E8M0 = 37,
    ACL_FLOAT6_E3M2 = 38,
    ACL_FLOAT6_E2M3 = 39,
    ACL_FLOAT4_E2M1 = 40,
    ACL_FLOAT4_E1M2 = 41,
} aclDataType;

typedef enum {
    ACL_FORMAT_UNDEFINED = -1,
    ACL_FORMAT_NCHW = 0,
    ACL_FORMAT_NHWC = 1,
    ACL_FORMAT_ND = 2,
    ACL_FORMAT_NC1HWC0 = 3,
    ACL_FORMAT_FRACTAL_Z = 4,
    ACL_FORMAT_NC1HWC0_C04 = 12,
    ACL_FORMAT_HWCN = 16,
    ACL_FORMAT_NDHWC = 27,
    ACL_FORMAT_FRACTAL_NZ = 29,
    ACL_FORMAT_NCDHW = 30,
    ACL_FORMAT_NDC1HWC0 = 32,
    ACL_FRACTAL_Z_3D = 33,
    ACL_FORMAT_NC = 35,
    ACL_FORMAT_NCL = 47,
} aclFormat;

#ifdef __cplusplus
}
#endif

namespace atb {
constexpr size_t DEFAULT_SVECTOR_SIZE = 64;

template <class T> class SVector {
public:
    T *begin() noexcept
    {
        if (heap_) {
            return &heap_[0];
        }
        return &storage_[0];
    }

    const T *begin() const noexcept
    {
        if (heap_) {
            return &heap_[0];
        }
        return &storage_[0];
    }

    T *end() noexcept
    {
        if (heap_) {
            return (&heap_[0]) + size_;
        }
        return (&storage_[0]) + size_;
    }

    const T *end() const noexcept
    {
        if (heap_) {
            return (&heap_[0]) + size_;
        }
        return (&storage_[0]) + size_;
    }

private:
    std::size_t capacity_ = 0;
    std::size_t size_ = 0;
    T storage_[DEFAULT_SVECTOR_SIZE + 1];
    T *heap_ = nullptr;
};

using Status = int32_t;
constexpr uint32_t MAX_DIM = 8;

struct Dims {
    int64_t dims[MAX_DIM];
    uint64_t dimNum = 0;
};

struct TensorDesc {
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclFormat format = ACL_FORMAT_UNDEFINED;
    Dims shape;
};

struct Tensor {
    TensorDesc desc;
    void *deviceData = nullptr;
    void *hostData = nullptr;
    uint64_t dataSize = 0;
};

class ContextBase {};

struct RunnerVariantPack {
    SVector<Tensor> inTensors;
    SVector<Tensor> outTensors;
    SVector<bool> isInTensorCanFree;
    SVector<bool> isOutTensorNeedMalloc;
    uint8_t *hostTilingBuffer = nullptr;
    uint8_t *tilingBuffer = nullptr;
    uint64_t tilingBufferSize = 0;
    uint8_t *workspaceBuffer = nullptr;
    uint64_t workspaceBufferSize = 0;
    uint8_t *intermediateBuffer = nullptr;
    uint64_t intermediateBufferSize = 0;
    ContextBase *context = nullptr;
};

class Runner {};
}

#endif