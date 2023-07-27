//
// Created by Zero on 2023/7/27.
//

#pragma once

#include "core/stl.h"

namespace ocarina {

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

}// namespace ocarina