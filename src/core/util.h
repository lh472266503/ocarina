//
// Created by Zero on 15/05/2022.
//

#pragma once

#include "basic_traits.h"
#include "concepts.h"

namespace katana {

template<typename T, typename U, KTN_REQUIRES(is_integral_v<T> &&is_integral_v<U>)>
KTN_NODISCARD static constexpr auto mem_offset(T offset, U alignment) noexcept {
    return (offset + alignment - 1u) / alignment * alignment;
}

template<typename T>
requires concepts::multiply_able<T>
    KTN_NODISCARD constexpr auto sqr(T v) {
    return v * v;
}

template<typename T, int n>
requires concepts::multiply_able<T>
    KTN_NODISCARD constexpr float Pow(T v) {
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

}// namespace katana