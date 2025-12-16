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

#ifndef STARS_COMMON_H
#define STARS_COMMON_H

#include <cstdint>

/*
 * 在2023年11月 task id底层存在规则变动，取用了stream id中某位作为标记位，基于标记位进行新的task id计算，具体计算规则为：
 * 1、判断stream id的第13位是否为0，为0则直接返回task id，否则
 * 2、取task id低13位，与stream id高3位，组成一个完整taskId
 *
 * 2024年7月 task id规则变动，新增stream id标记位，扩展新的task id计算规则，变更后如下：
 * 1、判断stream id的第13位是否为0
 * 2、不为0，则取task id低13位，与stream id高3位，组成一个完整taskId
 * 3、为0，则继续判断stream id第14位是否为0，是则直接返回task id；不为0 则交换stream id和task id的低12位

* 2025年12月 runtime对原有stream做了扩容方案，stream的上限由2k变为32k，stream id task id字段发生变动，因此对规则进行变更：
 * 新增expandStatus参数用于区分是否处于扩容状态，新增sqeType参数用于区分SQE类型(针对某些特别算子做的适配)，具体规则如下：
 *
 * A. 当expandStatus==1（扩容状态）时：
 *    按照新CANN扩容逻辑进行解析
 *    1、GetStreamId函数：
 *        a) 如果sqeType为PLACE_HOLDER_SQE(3)或EVENT_RECORD_SQE(4)，则返回streamId的低15位,
 *        b) 如果streamId的第16位为1，则返回taskId的低15位
 *        c) 否则返回streamId的低15位
 *    2、GetTaskId函数：
 *        a) 如果sqeType为PLACE_HOLDER_SQE(3)或EVENT_RECORD_SQE(4)，则直接返回taskId
 *        b) 如果streamId的第16位为1，则将taskId的第16位与streamId的低15位组合
 *        c) 否则直接返回taskId
 *
 * B. 当expandStatus!=1（非扩容状态）时：
 *    按照原CANN逻辑进行解析
 */

namespace MemScope {
const uint16_t STREAM_LOW_OPERATOR = 1u << 11;
const uint16_t STREAM_JUDGE_BIT12_OPERATOR = 0x1000;
const uint16_t STREAM_JUDGE_BIT13_OPERATOR = 0x2000;
const uint16_t STREAM_JUDGE_BIT15_OPERATOR = 0x8000;
const uint16_t TASK_LOW_OPERATOR = 0x1FFF;
const uint16_t STREAM_HIGH_OPERATOR = 0xE000;
const uint16_t COMMON_LOW_OPERATOR = 0x0FFF;
const uint16_t COMMON_HIGH_OPERATOR = 0xF000;
const uint16_t EXPANDING_LOW_OPERATOR = 0x7FFF;
const uint16_t PLACE_HOLDER_SQE = 3;
const uint16_t EVENT_RECORD_SQE = 4;

class StarsCommon {
public:
    static void SetStreamExpandStatus(uint8_t expandStatus);
    static uint16_t GetStreamId(uint16_t streamId, uint16_t taskId, uint16_t sqeType = 0);
    static uint16_t GetTaskId(uint16_t streamId, uint16_t taskId, uint16_t sqeType = 0);
private:
    static bool isExpand;
};

} // namespace MemScope
#endif
