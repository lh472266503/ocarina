//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "header.h"
#include <functional>
#include <deque>
#include <queue>
#include <variant>
#include <span>
#include <vector>
#include <map>
#include <set>
#include <array>
#include <list>
#include <optional>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <string>
#include <unordered_set>
#include <string_view>
#include <filesystem>
#include <EASTL/tuple.h>
#include <EASTL/string.h>
#include <numeric>

#define OC_FORWARD(arg) std::forward<decltype(arg)>(arg)

namespace ocarina {

namespace detail {
OC_CORE_API void *allocator_allocate(size_t size, size_t alignment) noexcept;
OC_CORE_API void allocator_deallocate(void *p, size_t alignment) noexcept;
OC_CORE_API void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept;
}// namespace detail

template<typename T = std::byte>
struct allocator {
    using value_type = T;
    constexpr allocator() noexcept = default;
    template<typename U>
    constexpr explicit allocator(allocator<U>) noexcept {}
    [[nodiscard]] auto allocate(std::size_t n) const noexcept {
        return static_cast<T *>(detail::allocator_allocate(sizeof(T) * n, alignof(T)));
    }
    void deallocate(T *p, size_t) const noexcept {
        detail::allocator_deallocate(p, alignof(T));
    }
    template<typename R>
    [[nodiscard]] constexpr bool operator==(allocator<R>) const noexcept {
        return std::is_same_v<T, R>;
    }
};

template<typename T = std::byte>
[[nodiscard]] inline auto allocate(size_t n = 1u) noexcept {
    return allocator<T>{}.allocate(n);
}

template<typename T>
inline void deallocate(T *p) noexcept {
    using type = std::remove_cvref_t<T>;
    allocator<type>{}.deallocate(const_cast<type *>(p), 0u);
}

template<typename T, typename... Args>
[[nodiscard]] inline auto new_with_allocator(Args &&...args) noexcept {
    return std::construct_at(allocate<T>(), std::forward<Args>(args)...);
}

template<typename T>
inline void delete_with_allocator(T *p) noexcept {
    if (p != nullptr) {
        std::destroy_at(p);
        deallocate(p);
    }
}

template<typename T = std::byte>
OC_NODISCARD T *new_array(size_t num) noexcept {
    return new T[num];
}

template<typename T>
void delete_array(T *ptr) noexcept {
    delete[] ptr;
}

// io
using std::cout;
using std::endl;

// ptr
using std::const_pointer_cast;
using std::dynamic_pointer_cast;
using std::enable_shared_from_this;
using std::function;
using std::make_shared;
using std::make_unique;
using std::reinterpret_pointer_cast;
using std::shared_ptr;
using std::static_pointer_cast;
using std::unique_ptr;
using std::weak_ptr;

template<typename T>
using UP = unique_ptr<T>;
template<typename T>
using SP = shared_ptr<T>;

// math
using std::abs;
using std::atan2;
using std::max;
using std::min;
using std::pow;
using std::sqrt;
using std::sin;
using std::cos;
using std::tan;
using std::asin;
using std::acos;
using std::atan;

inline void oc_memcpy(void *dst, const void *src, size_t size) {
#ifdef _MSC_VER
    std::memcpy(dst, src, size);
#else
    std::wmemcpy(reinterpret_cast<wchar_t *>(dst), reinterpret_cast<const wchar_t *>(src), size);
#endif
}

struct MemoryBlock {
public:
    const void *address{};
    size_t size{};
    size_t alignment{};

    // todo calculate max member size by object
    size_t max_member_size = 8;

public:
    MemoryBlock() = default;
    MemoryBlock(const void *address, size_t size,
                size_t alignment, size_t max_member_size)
        : address(address), size(size),
          alignment(alignment), max_member_size(max_member_size) {}
};

// string
using std::string;
using std::string_view;
using std::to_string;

template<class To, class From>
requires(sizeof(To) == sizeof(From) &&
         std::is_trivially_copyable_v<From> &&
         std::is_trivially_copyable_v<To>)
    [[nodiscard]] To bit_cast(const From &src) noexcept {
    static_assert(std::is_trivially_constructible_v<To>,
                  "This implementation requires the destination type to be trivially "
                  "constructible");
    union {
        From from;
        To to;
    } u;
    u.from = src;
    return u.to;
}

inline size_t substr_count(string_view str, string_view target) noexcept {
    string::size_type pos = 0;
    size_t ret = 0;
    while ((pos = str.find(target, pos)) != string::npos) {
        ret += 1;
        pos += target.length();
    }
    return ret;
}

// range and container
using std::array;
using std::deque;
using std::list;
using std::map;
using std::optional;
using std::queue;
using std::set;
using std::span;
using std::unordered_map;
using std::unordered_set;
using std::vector;

#if 1
// tuple
using eastl::get;
using eastl::tuple;
using eastl::tuple_element;
using eastl::tuple_element_t;
using eastl::tuple_size;
using eastl::tuple_size_v;
#else
using std::get;
using std::tuple;
using std::tuple_element;
using std::tuple_element_t;
using std::tuple_size;
using std::tuple_size_v;
#endif
// sequence
using std::index_sequence;
using std::index_sequence_for;
using std::integer_sequence;
using std::make_index_sequence;
using std::make_integer_sequence;

// other
using std::monostate;
using std::pair;
using std::variant;
using std::visit;
using std::make_pair;
using std::function;
namespace fs = std::filesystem;

}// namespace ocarina