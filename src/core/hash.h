//
// Created by Zero on 29/04/2022.
//

#pragma once

#include "xxhash.h"
#include "stl.h"

namespace sycamore {
namespace detail {

SCM_NODISCARD inline auto xxh3_hash64(const void *data, size_t size, uint64_t seed) noexcept {
    return XXH3_64bits_withSeed(data, size, seed);
}

template<typename T>
concept hashable_with_hash_method = requires(T x) {
    x.hash();
};

}// namespace detail

SCM_NODISCARD SCM_CORE_API std::string_view hash_to_string(uint64_t hash) noexcept;

class Hash64 {
public:
    static constexpr auto default_seed = 19980810ull;

private:
    uint64_t _seed;

public:
    /// Constructor, set seed
    explicit constexpr Hash64(uint64_t seed = default_seed) noexcept
        : _seed{seed} {}

    template<typename T>
    SCM_NODISCARD auto operator()(T &&s) const noexcept -> uint64_t {
        if constexpr (sycamore::detail::hashable_with_hash_method<T>) {
        }
    }
};

}// namespace sycamore
