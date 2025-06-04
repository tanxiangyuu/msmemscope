// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef UTILITY_SERIALIZER_H
#define UTILITY_SERIALIZER_H

#include <string>
#include <type_traits>
#include <algorithm>

namespace Leaks {

template<typename T, typename = typename std::enable_if<std::is_pod<T>::value>::type>
inline std::string Serialize(const T &val)
{
    constexpr std::size_t size = sizeof(T);
    std::string msg(static_cast<char const *>(static_cast<void const *>(&val)), size);
    return msg;
}

template<typename T, typename... Ts, typename = typename std::enable_if<std::is_pod<T>::value>::type>
inline std::string Serialize(const T &val, const Ts &... vals)
{
    return Serialize(val) + Serialize(vals...);
}

template<typename T, typename = typename std::enable_if<std::is_pod<T>::value>::type>
inline bool Deserialize(const std::string &msg, T &val)
{
    constexpr std::size_t size = sizeof(T);
    if (msg.size() < size) {
        return false;
    }
    std::copy_n(msg.data(), size, static_cast<char *>(static_cast<void *>(&val)));
    return true;
}

}

#endif
