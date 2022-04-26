//
// Created by Zero on 25/04/2022.
//

#pragma once

#include "basic_trait.h"

namespace sycamore {
inline namespace size_literals {

SCM_NODISCARD constexpr auto operator""_kb(unsigned long long bytes) noexcept {
    return static_cast<size_t>(bytes * 1024u);
}

SCM_NODISCARD constexpr auto operator""_mb(unsigned long long bytes) noexcept {
    return static_cast<size_t>(bytes * 1024u * 1024u);
}

SCM_NODISCARD constexpr auto operator""_gb(unsigned long long bytes) noexcept {
    return static_cast<size_t>(bytes * 1024u * 1024u * 1024u);
}

//template<typename T, size_t N>
//struct VectorStorage {
//    static_assert(false, "Invalid vector storage");
//};


//template<typename T>
//struct alignas(sizeof(T) * 2): VectorStorage<T, 2> {
//
//}

}



}// namespace sycamore::core