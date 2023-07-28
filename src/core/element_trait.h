//
// Created by Zero on 2023/7/27.
//

#pragma once

#include "core/stl.h"

namespace ocarina {

template<typename... T>
struct always_false : std::false_type {};

template<typename... T>
constexpr auto always_false_v = always_false<T...>::value;

template<typename... T>
struct always_true : std::true_type {};

template<typename... T>
constexpr auto always_true_v = always_true<T...>::value;

namespace detail {

template<typename T>
struct element_impl {
    using type = T;
};

template<typename T>
struct element_impl<vector<T>> {
    using type = T;
};

template<typename T>
struct element_impl<list<T>> {
    using type = T;
};

template<typename T>
struct element_impl<std::stack<T>> {
    using type = T;
};

template<typename T>
struct element_impl<std::deque<T>> {
    using type = T;
};

template<typename T>
struct element_impl<std::queue<T>> {
    using type = T;
};

template<typename T>
struct element_impl<unique_ptr<T>> {
    using type = T;
};

template<typename T>
struct element_impl<shared_ptr<T>> {
    using type = T;
};

}// namespace detail

template<typename T>
using element_t = detail::element_impl<std::remove_cvref_t<T>>::type;

namespace detail {
template<typename T>
struct ptr_impl {
    static_assert(always_false_v<T>);
    using type = T;
};

template<typename T>
struct ptr_impl<T *> {
    using type = std::remove_cvref_t<T>;
};

template<typename T>
struct ptr_impl<unique_ptr<T>> {
    using type = T;
};

template<typename T>
struct ptr_impl<shared_ptr<T>> {
    using type = T;
};

template<typename T>
struct is_ptr_impl : std::false_type {};

template<typename T>
struct is_ptr_impl<T *> : std::true_type {};

template<typename T>
struct is_ptr_impl<unique_ptr<T>> : std::true_type {};

template<typename T>
struct is_ptr_impl<shared_ptr<T>> : std::true_type {};

}// namespace detail

template<typename T>
using ptr_t = detail::ptr_impl<std::remove_cvref_t<T>>::type;

template<typename T>
static constexpr bool is_ptr_v = detail::is_ptr_impl<std::remove_cvref_t<T>>::value;

template<typename Arg>
requires is_ptr_v<Arg>
[[nodiscard]] ptr_t<Arg> *raw_ptr(Arg arg) {
    if constexpr (std::is_pointer_v<Arg>) {
        return arg;
    } else {
        return arg.get();
    }
}

}// namespace ocarina