//
// Created by Zero on 15/05/2022.
//

#pragma once

#include "basic_traits.h"
#include "concepts.h"

namespace katana {

template<typename T, typename U, KTN_REQUIRES(is_integral_v<T> &&is_integral_v<U>)>
[[nodiscard]] static constexpr auto mem_offset(T offset, U alignment) noexcept {
    return (offset + alignment - 1u) / alignment * alignment;
}

template<typename T>
requires concepts::multiply_able<T>
    [[nodiscard]] constexpr auto sqr(T v) {
    return v * v;
}

template<int n, typename T>
requires concepts::multiply_able<T>
    [[nodiscard]] constexpr T Pow(T v) {
    if constexpr (n < 0) {
        return 1.f / Pow<-n>(v);
    } else if constexpr (n == 1) {
        return v;
    } else if constexpr (n == 0) {
        return 1;
    }
    float n2 = Pow<n / 2>(v);
    return n2 * n2 * Pow<n & 1>(v);
}

inline namespace size_literals {
    [[nodiscard]] constexpr auto operator""_kb(size_t bytes) noexcept {
    return static_cast<size_t>(bytes * 1024u);
}

[[nodiscard]] constexpr auto operator""_mb(size_t bytes) noexcept {
    return static_cast<size_t>(bytes * sqr(1024u));
}

[[nodiscard]] constexpr auto operator""_gb(size_t bytes) noexcept {
    return static_cast<size_t>(bytes * Pow<3>(1024u));
}
}// namespace size_literals

}// namespace katana