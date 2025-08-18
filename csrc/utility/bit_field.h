// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

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