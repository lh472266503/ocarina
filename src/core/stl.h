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

template<typename T>
[[nodiscard]] inline auto allocate(size_t n = 1u) noexcept {
    return allocator<T>{}.allocate(n);
}

template<typename T>
inline void deallocate(T *p) noexcept {
    allocator<T>{}.deallocate(p, 0u);
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

// string
using std::string;
using std::string_view;
using std::to_string;

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
namespace fs = std::filesystem;

}// namespace ocarina