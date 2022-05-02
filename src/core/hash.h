//
// Created by Zero on 29/04/2022.
//

#pragma once
//#define XXH_STATIC_LINKING_ONLY   /* access advanced declarations */
//#define XXH_INLINE_ALL   /* access definitions */
#include "xxhash.h"
#include "stl.h"
#include "basic_types.h"
#include "concepts.h"

namespace sycamore {
namespace detail {

SCM_NODISCARD inline auto xxh3_hash64(const void *data, size_t size, uint64_t seed) noexcept {
    return XXH3_64bits_withSeed(data, size, seed);
}

template<typename T>
concept hashable_with_hash_method = requires(T x) {
    x.hash();
};

template<typename T>
concept hashable_with_hash_code_method = requires(T x) {
    x.hash_code();
};

}// namespace detail

SCM_NODISCARD SCM_CORE_API std::string_view hash_to_string(uint64_t hash) noexcept;

class Hash64 {
public:
    static constexpr auto default_seed = 123456789ull;

private:
    uint64_t _seed;

public:
    /// Constructor, set seed
    explicit constexpr Hash64(uint64_t seed = default_seed) noexcept
        : _seed{seed} {}

    template<typename T>
    SCM_NODISCARD uint64_t operator()(T &&s) const noexcept {
        if constexpr (sycamore::detail::hashable_with_hash_method<T>) {
            return (*this)(std::forward<T>(s).hash());
        } else if constexpr (detail::hashable_with_hash_code_method<T>) {
            return (*this)(std::forward<T>(s).hash_code());
        } else if constexpr (concepts::string_viewable<T>) {
            std::string_view sv{std::forward<T>(s)};
            return detail::xxh3_hash64(sv.data(), sv.size(), _seed);
        } else if constexpr (is_vector3_v<T>) {
            auto x = s;
            return detail::xxh3_hash64(&x, sizeof(vector_element_t<T>) * 3u, _seed);
        } else if constexpr (is_matrix3_v<T>) {
            auto x = sycamore::make_float4x4(s);
            return (*this)(x);
        } else if constexpr (
            std::is_arithmetic_v<std::remove_cvref_t<T>> ||
            std::is_enum_v<std::remove_cvref_t<T>> ||
            is_basic_v<T>) {
            auto x = s;
            return detail::xxh3_hash64(&x, sizeof(x), _seed);
        } else {
            static_assert(always_false_v<T>);
        }
    }
};

template<typename T>
SCM_NODISCARD inline uint64_t hash64(T &&v, uint64_t seed = Hash64::default_seed) noexcept {
    return Hash64{seed}(std::forward<T>(v));
}

}// namespace sycamore
