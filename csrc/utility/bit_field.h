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

#ifndef BIT_FIELD_H
#define BIT_FIELD_H

#include <type_traits>
#include <climits>

template <typename T>
class BitField {
public:
    BitField(T init = 0) : value(init) {}
    
    void setBit(size_t bit)
    {
        if (bit < sizeof(T) * CHAR_BIT) {
            value |= (static_cast<T>(1) << bit);
        }
    }
    
    bool checkBit(size_t bit) const
    {
        if (bit < sizeof(T) * CHAR_BIT) {
            return (value & (static_cast<T>(1) << bit)) != 0;
        }
        return false;
    }
    
    T getValue() const
    {
        return value;
    }
    
private:
    T value;
};

template <typename T>
bool BitPresent(T value, size_t bit)
{
    return BitField<T>(value).checkBit(bit);
}

#endif