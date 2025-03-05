// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef MODULE_INFO_H
#define MODULE_INFO_H


namespace Leaks {

// Module id
const std::unordered_map<int, std::string> MODULE_HASH_TABLE = {
    {0, "SLOG"},          /**< Slog */
    {1, "IDEDD"},         /**< IDE daemon device */
    {2, "IDEDH"},         /**< IDE daemon host */
    {3, "HCCL"},          /**< HCCL */
    {4, "FMK"},           /**< Adapter */
    {5, "HIAIENGINE"},    /**< Matrix */
    {6, "DVPP"},          /**< DVPP */
    {7, "RUNTIME"},       /**< Runtime */
    {8, "CCE"},           /**< CCE */
    {9, "HDC"},           /**< HDC */
    {10, "DRV"},           /**< Driver */
    {11, "MDCFUSION"},     /**< Mdc fusion */
    {12, "MDCLOCATION"},   /**< Mdc location */
    {13, "MDCPERCEPTION"}, /**< Mdc perception */
    {14, "MDCFSM"},
    {15, "MDCCOMMON"},
    {16, "MDCMONITOR"},
    {17, "MDCBSWP"},    /**< MDC base software platform */
    {18, "MDCDEFAULT"}, /**< MDC undefine */
    {19, "MDCSC"},      /**< MDC spatial cognition */
    {20, "MDCPNC"},
    {21, "MLL"},      /**< abandon */
    {22, "DEVMM"},    /**< Dlog memory managent */
    {23, "KERNEL"},   /**< Kernel */
    {24, "LIBMEDIA"}, /**< Libmedia */
    {25, "CCECPU"},   /**< aicpu shedule */
    {26, "ASCENDDK"}, /**< AscendDK */
    {27, "ROS"},      /**< ROS */
    {28, "HCCP"},
    {29, "ROCE"},
    {30, "TEFUSION"},
    {31, "PROFILING"}, /**< Profiling */
    {32, "DP"},        /**< Data Preprocess */
    {33, "APP"},       /**< User Application */
    {34, "TS"},        /**< TS module */
    {35, "TSDUMP"},    /**< TSDUMP module */
    {36, "AICPU"},     /**< AICPU module */
    {37, "LP"},        /**< LP module */
    {38, "TDT"},       /**< tsdaemon or aicpu shedule */
    {39, "FE"},
    {40, "MD"},
    {41, "MB"},
    {42, "ME"},
    {43, "IMU"},
    {44, "IMP"},
    {45, "GE"}, /**< Fmk */
    {46, "MDCFUSA"},
    {47, "CAMERA"},
    {48, "ASCENDCL"},
    {49, "TEEOS"},
    {50, "ISP"},
    {51, "SIS"},
    {52, "HSM"},
    {53, "DSS"},
    {54, "PROCMGR"},     // Process Manager, Base Platform
    {55, "BBOX"},
    {56, "AIVECTOR"},
    {57, "TBE"},
    {58, "FV"},
    {59, "MDCMAP"},
    {60, "TUNE"},
    {61, "HSS"}, /**< helper */
    {62, "FFTS"},
    {63, "OP"},
    {64, "UDF"},
    {65, "HICAID"},
    {66, "TSYNC"},
    {67, "AUDIO"},
    {68, "TPRT"},
    {69, "ASCENDCKERNEL"},
    {70, "ASYS"},
    {71, "ATRACE"},
    {72, "RTC"},
    {73, "SYSMONITOR"},
    {74, "AML"},
    {75, "INVLID_MOUDLE_ID"}    // add new module before INVLID_MOUDLE_ID
};

}

#endif