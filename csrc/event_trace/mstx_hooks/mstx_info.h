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

#ifndef MSTX_INFO_H
#define MSTX_INFO_H


typedef enum mstxFuncModule {
    MSTX_API_MODULE_INVALID = 0,
    MSTX_API_MODULE_CORE = 1,
    MSTX_API_MODULE_CORE_DOMAIN = 2,
    MSTX_API_MODULE_CORE_MEM = 3,
    MSTX_API_MODULE_SIZE,  // end of the enum, new enum items must be added before this
    MSTX_API_MODULE_FORCE_INT = 0x7fffffff
} mstxFuncModule;

typedef enum mstxImplCoreMemFuncId {
    MSTX_API_CORE_MEM_INVALID               = 0,
    MSTX_API_CORE_MEMHEAP_REGISTER          = 1,
    MSTX_API_CORE_MEMHEAP_UNREGISTER        = 2,
    MSTX_API_CORE_MEM_REGIONS_REGISTER      = 3,
    MSTX_API_CORE_MEM_REGIONS_UNREGISTER    = 4,
    MSTX_API_CORE_MEM_SIZE,                   // end of the enum, new enum items must be added before this
    MSTX_API_CORE_MEM_FORCE_INT             = 0x7fffffff
} mstxImplCoreMemFuncId;

typedef enum mstxImplCoreFuncId {
    MSTX_API_CORE_INVALID                   = 0,
    MSTX_API_CORE_MARK_A                    = 1,
    MSTX_API_CORE_RANGE_START_A             = 2,
    MSTX_API_CORE_RANGE_END                 = 3,
    MSTX_API_CORE_SIZE,                   // end of the enum, new enum items must be added before this
    MSTX_API_CORE_FORCE_INT = 0x7fffffff
} mstxImplCoreFuncId;

typedef enum mstxImplCoreDomainFuncId {
    MSTX_API_CORE2_INVALID                 =  0,
    MSTX_API_CORE2_DOMAIN_CREATE_A         =  1,
    MSTX_API_CORE2_DOMAIN_DESTROY          =  2,
    MSTX_API_CORE2_DOMAIN_MARK_A           =  3,
    MSTX_API_CORE2_DOMAIN_RANGE_START_A    =  4,
    MSTX_API_CORE2_DOMAIN_RANGE_END        =  5,
    MSTX_API_CORE2_SIZE,                   // end of the enum, new enum items must be added before this
    MSTX_API_CORE2_FORCE_INT = 0x7fffffff
} mstxImplCoreDomainFuncId;

struct mstxDomainRegistration_st {};
typedef struct mstxDomainRegistration_st mstxDomainRegistration_t;
typedef mstxDomainRegistration_t* mstxDomainHandle_t;

struct mstxMemHeap_st {};
typedef struct mstxMemHeap_st mstxMemHeap_t;
typedef mstxMemHeap_t* mstxMemHeapHandle_t;

struct mstxMemRegion_st {};
typedef struct mstxMemRegion_st mstxMemRegion_t;
typedef mstxMemRegion_t* mstxMemRegionHandle_t;

typedef struct mstxMemVirtualRangeDesc_t {
    uint32_t deviceId;
    void const *ptr;
    int64_t size;
} mstxMemVirtualRangeDesc_t;


typedef enum mstxMemHeapUsageType {
    MSTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR = 0,
} mstxMemHeapUsageType;


typedef enum mstxMemType {
    MSTX_MEM_TYPE_VIRTUAL_ADDRESS = 0,
} mstxMemType;

typedef struct mstxMemHeapDesc_t {
    mstxMemHeapUsageType usage;
    mstxMemType type;
    void const *typeSpecificDesc;
} mstxMemHeapDesc_t;

typedef struct mstxMemRegionsRegisterBatch_t {
    mstxMemHeapHandle_t heap;
    mstxMemType regionType;
    size_t regionCount;
    void const *regionDescArray;
    mstxMemRegionHandle_t* regionHandleArrayOut;
} mstxMemRegionsRegisterBatch_t;


typedef enum mstxMemRegionRefType {
    MSTX_MEM_REGION_REF_TYPE_POINTER = 0,
    MSTX_MEM_REGION_REF_TYPE_HANDLE
} mstxMemRegionRefType;

typedef struct mstxMemRegionRef_t {
    mstxMemRegionRefType refType;
    union {
        void const *pointer;
        mstxMemRegionHandle_t handle;
    };
} mstxMemRegionRef_t;

typedef struct mstxMemRegionsUnregisterBatch_t {
    size_t refCount;
    mstxMemRegionRef_t const *refArray;
} mstxMemRegionsUnregisterBatch_t;

#endif