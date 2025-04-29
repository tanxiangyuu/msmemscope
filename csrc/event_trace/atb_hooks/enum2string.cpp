// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "enum2string.h"

#include <unordered_map>

namespace atb {
    const std::string& LeaksEnumToString(aclDataType value)
    {
        static std::unordered_map<aclDataType, std::string> aclDataTypeMap = {
            {ACL_DT_UNDEFINED, "ACL_DT_UNDEFINED"},
            {ACL_FLOAT, "ACL_FLOAT"},
            {ACL_FLOAT16, "ACL_FLOAT16"},
            {ACL_INT8, "ACL_INT8"},
            {ACL_INT32, "ACL_INT32"},
            {ACL_UINT8, "ACL_UINT8"},
            {ACL_INT16, "ACL_INT16"},
            {ACL_UINT16, "ACL_UINT16"},
            {ACL_UINT32, "ACL_UINT32"},
            {ACL_INT64, "ACL_INT64"},
            {ACL_UINT64, "ACL_UINT64"},
            {ACL_DOUBLE, "ACL_DOUBLE"},
            {ACL_BOOL, "ACL_BOOL"},
            {ACL_STRING, "ACL_STRING"},
            {ACL_COMPLEX64, "ACL_COMPLEX64"},
            {ACL_COMPLEX128, "ACL_COMPLEX128"},
            {ACL_BF16, "ACL_BF16"},
            {ACL_INT4, "ACL_INT4"},
            {ACL_UINT1, "ACL_UINT1"},
            {ACL_COMPLEX32, "ACL_COMPLEX32"},
        };
        auto it = aclDataTypeMap.find(value);
        if (it != aclDataTypeMap.end()) {
            return it->second;
        } else {
            static const std::string UNDEFINED = "ACL_DT_UNDEFINED";
            return UNDEFINED;
        }
    }

    const std::string& LeaksEnumToString(aclFormat value)
    {
        static std::unordered_map<aclFormat, std::string> aclFormatMap = {
            {ACL_FORMAT_UNDEFINED, "ACL_FORMAT_UNDEFINED"},
            {ACL_FORMAT_NCHW, "ACL_FORMAT_NCHW"},
            {ACL_FORMAT_NHWC, "ACL_FORMAT_NHWC"},
            {ACL_FORMAT_ND, "ACL_FORMAT_ND"},
            {ACL_FORMAT_NC1HWC0, "ACL_FORMAT_NC1HWC0"},
            {ACL_FORMAT_FRACTAL_Z, "ACL_FORMAT_FRACTAL_Z"},
            {ACL_FORMAT_NC1HWC0_C04, "ACL_FORMAT_NC1HWC0_C04"},
            {ACL_FORMAT_HWCN, "ACL_FORMAT_HWCN"},
            {ACL_FORMAT_NDHWC, "ACL_FORMAT_NDHWC"},
            {ACL_FORMAT_FRACTAL_NZ, "ACL_FORMAT_FRACTAL_NZ"},
            {ACL_FORMAT_NCDHW, "ACL_FORMAT_NCDHW"},
            {ACL_FORMAT_NDC1HWC0, "ACL_FORMAT_NDC1HWC0"},
            {ACL_FRACTAL_Z_3D, "ACL_FRACTAL_Z_3D"},
            {ACL_FORMAT_NC, "ACL_FORMAT_NC"},
            {ACL_FORMAT_NCL, "ACL_FORMAT_NCL"},
        };
        auto it = aclFormatMap.find(value);
        if (it != aclFormatMap.end()) {
            return it->second;
        } else {
            static const std::string UNDEFINED = "ACL_FORMAT_UNDEFINED";
            return UNDEFINED;
        }
    }

    const std::string& LeaksEnumToString(Mki::TensorDType value)
    {
        static std::unordered_map<Mki::TensorDType, std::string> tensorDTypeMap = {
            {Mki::TENSOR_DTYPE_UNDEFINED, "TENSOR_DTYPE_UNDEFINED"},
            {Mki::TENSOR_DTYPE_FLOAT, "TENSOR_DTYPE_FLOAT"},
            {Mki::TENSOR_DTYPE_FLOAT16, "TENSOR_DTYPE_FLOAT16"},
            {Mki::TENSOR_DTYPE_INT8, "TENSOR_DTYPE_INT8"},
            {Mki::TENSOR_DTYPE_INT32, "TENSOR_DTYPE_INT32"},
            {Mki::TENSOR_DTYPE_UINT8, "TENSOR_DTYPE_UINT8"},
            {Mki::TENSOR_DTYPE_INT16, "TENSOR_DTYPE_INT16"},
            {Mki::TENSOR_DTYPE_UINT16, "TENSOR_DTYPE_UINT16"},
            {Mki::TENSOR_DTYPE_UINT32, "TENSOR_DTYPE_UINT32"},
            {Mki::TENSOR_DTYPE_INT64, "TENSOR_DTYPE_INT64"},
            {Mki::TENSOR_DTYPE_UINT64, "TENSOR_DTYPE_UINT64"},
            {Mki::TENSOR_DTYPE_DOUBLE, "TENSOR_DTYPE_DOUBLE"},
            {Mki::TENSOR_DTYPE_BOOL, "TENSOR_DTYPE_BOOL"},
            {Mki::TENSOR_DTYPE_STRING, "TENSOR_DTYPE_STRING"},
            {Mki::TENSOR_DTYPE_COMPLEX64, "TENSOR_DTYPE_COMPLEX64"},
            {Mki::TENSOR_DTYPE_COMPLEX128, "TENSOR_DTYPE_COMPLEX128"},
            {Mki::TENSOR_DTYPE_BF16, "TENSOR_DTYPE_BF16"},
        };
        auto it = tensorDTypeMap.find(value);
        if (it != tensorDTypeMap.end()) {
            return it->second;
        } else {
            static const std::string UNDEFINED = "TENSOR_DTYPE_UNDEFINED";
            return UNDEFINED;
        }
    }

    const std::string& LeaksEnumToString(Mki::TensorFormat value)
    {
        static std::unordered_map<Mki::TensorFormat, std::string> tensorFormatMap = {
            {Mki::TENSOR_FORMAT_UNDEFINED, "TENSOR_FORMAT_UNDEFINED"},
            {Mki::TENSOR_FORMAT_NCHW, "TENSOR_FORMAT_NCHW"},
            {Mki::TENSOR_FORMAT_NHWC, "TENSOR_FORMAT_NHWC"},
            {Mki::TENSOR_FORMAT_ND, "TENSOR_FORMAT_ND"},
            {Mki::TENSOR_FORMAT_NC1HWC0, "TENSOR_FORMAT_NC1HWC0"},
            {Mki::TENSOR_FORMAT_FRACTAL_Z, "TENSOR_FORMAT_FRACTAL_Z"},
            {Mki::TENSOR_FORMAT_NC1HWC0_C04, "TENSOR_FORMAT_NC1HWC0_C04"},
            {Mki::TENSOR_FORMAT_HWCN, "TENSOR_FORMAT_HWCN"},
            {Mki::TENSOR_FORMAT_NDHWC, "TENSOR_FORMAT_NDHWC"},
            {Mki::TENSOR_FORMAT_FRACTAL_NZ, "TENSOR_FORMAT_FRACTAL_NZ"},
            {Mki::TENSOR_FORMAT_NCDHW, "TENSOR_FORMAT_NCDHW"},
            {Mki::TENSOR_FORMAT_NDC1HWC0, "TENSOR_FORMAT_NDC1HWC0"},
            {Mki::TENSOR_FORMAT_FRACTAL_Z_3D, "TENSOR_FORMAT_FRACTAL_Z_3D"},
        };
        auto it = tensorFormatMap.find(value);
        if (it != tensorFormatMap.end()) {
            return it->second;
        } else {
            static const std::string UNDEFINED = "TENSOR_FORMAT_UNDEFINED";
            return UNDEFINED;
        }
    }
}