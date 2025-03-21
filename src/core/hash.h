//
// Created by Zero on 29/04/2022.
//

#pragma once
//#define XXH_STATIC_LINKING_ONLY   /* access advanced declarations */
//#define XXH_INLINE_ALL   /* access definitions */
#include "xxhash.h"
#include "stl.h"
#include "math/basic_types.h"
#include "concepts.h"

namespace ocarina {
namespace detail {

template<typename T>
concept hashable_with_hash_method = requires(T x) {
    x.hash();
};

template<typename T>
concept hashable_ptr_with_hash_method = requires(T x) {
    x->hash();
};

template<typename T>
concept hashable_with_hash_code_method = requires(T x) {
    x.hash_code();
};

}// namespace detail

[[nodiscard]] inline auto xxh3_hash64(const void *data, size_t size, uint64_t seed) noexcept {
    return XXH3_64bits_withSeed(data, size, seed);
}

namespace {
template<typename T>
struct is_3row_matrix {
    static constexpr bool value = false;
};

template<size_t N>
struct is_3row_matrix<Matrix<N, 3>> {
    static constexpr bool value = true;
};

template<typename T>
static constexpr bool is_3row_matrix_v = is_3row_matrix<std::remove_cvref_t<T>>::value;
}// namespace

[[nodiscard]] OC_CORE_API std::string_view hash_to_string(uint64_t hash) noexcept;

class Hash64 {
public:
    static constexpr auto default_seed = 123456789ull;

private:
    uint64_t seed_;

public:
    /// Constructor, set seed
    explicit constexpr Hash64(uint64_t seed = default_seed) noexcept
        : seed_{seed} {}

    template<typename T>
    [[nodiscard]] uint64_t operator()(T &&s) const noexcept {
        if constexpr (ocarina::detail::hashable_with_hash_method<T>) {
            return (*this)(std::forward<T>(s).hash());
        } else if constexpr (detail::hashable_ptr_with_hash_method<T>) {
            return (*this)(std::forward<T>(s)->hash());
        } else if constexpr (concepts::string_viewable<T>) {
            std::string_view sv{std::forward<T>(s)};
            return xxh3_hash64(sv.data(), sv.size(), seed_);
        } else if constexpr (is_vector3_v<T>) {
            auto x = s;
            return xxh3_hash64(&x, sizeof(vector_element_t<T>) * 3u, seed_);
        } else if constexpr (is_3row_matrix_v<T>) {
            uint64t ret = seed_;
            for (int i = 0; i < std::remove_cvref_t<T>::col_num; ++i) {
                ret = Hash64{ret}((*this)(s[i]));
            }
            return ret;
        } else if constexpr (
            std::is_standard_layout_v<std::remove_cvref_t<T>> ||
            std::is_arithmetic_v<std::remove_cvref_t<T>> ||
            std::is_enum_v<std::remove_cvref_t<T>> ||
            is_basic_v<std::remove_cvref_t<T>>) {
            auto x = s;
            return xxh3_hash64(&x, sizeof(x), seed_);
        } else {
            static_assert(always_false_v<T>);
            return {};
        }
    }
};

namespace detail {

template<typename T>
[[nodiscard]] inline uint64_t hash64(T &&v, uint64_t seed = Hash64::default_seed) noexcept {
    return Hash64{seed}(std::forward<T>(v));
}

}// namespace detail

template<typename... Args>
[[nodiscard]] uint64_t hash64(Args &&...args) noexcept {
    static constexpr auto size = sizeof...(args);
    array<uint64_t, size> arr = {detail::hash64(OC_FORWARD(args))...};
    uint64_t ret = Hash64::default_seed;
    for (int i = 0; i < size; ++i) {
        ret = detail::hash64(arr[i], ret);
    }
    return ret;
}

template<typename T>
requires concepts::iterable<T>
[[nodiscard]] uint64_t hash64_list(T &&lst) noexcept {
    size_t size = lst.size();
    uint64_t ret = Hash64::default_seed;
    for (const auto &elm : OC_FORWARD(lst)) {
        ret = detail::hash64(elm, ret);
    }
    return ret;
}

class RTTI {
public:
    [[nodiscard]] virtual const char *class_name() const noexcept {
        return typeid(*this).name();
    }
};

class Hashable : public RTTI {
private:
    mutable uint64_t _hash{0u};
    mutable bool _hash_computed{false};
    mutable uint64_t _type_hash{0u};
    mutable bool _type_hash_computed{false};

protected:
    [[nodiscard]] virtual uint64_t _compute_hash() const noexcept {
        return hash64(class_name(), reinterpret_cast<uint64_t>(this));
    }
    [[nodiscard]] virtual uint64_t _compute_type_hash() const noexcept {
        return hash64(class_name());
    }

public:
    void reset_hash() const noexcept { _hash_computed = false; }
    void reset_type_hash() const noexcept { _type_hash_computed = false; }

    [[nodiscard]] uint64_t hash() const noexcept {
        if (!_hash_computed) {
            _hash = hash64(class_name(), _compute_hash());
            _hash_computed = true;
        }
        return _hash;
    }

    [[nodiscard]] uint64_t type_hash() const noexcept {
        if (!_type_hash_computed) {
            _type_hash = hash64(class_name(), _compute_type_hash());
            _type_hash_computed = true;
        }
        return _type_hash;
    }

    template<typename T>
    static uint64_t compute_hash(uint64_t hash) {
        return hash64(typeid(std::remove_cvref_t<T>).name(), hash);
    }
    ~Hashable() = default;
};

}// namespace ocarina
