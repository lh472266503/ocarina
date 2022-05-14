//
// Created by Zero on 15/05/2022.
//

#pragma once

#include "basic_traits.h"

namespace katana {

template<typename T, typename U>
requires is_integral_v<T> && is_integral_v<U>
    KTN_NODISCARD static constexpr auto mem_offset(T offset, U alignment) noexcept {
    return (offset + alignment - 1u) / alignment * alignment;
}

}// namespace katana