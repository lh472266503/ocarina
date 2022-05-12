//
// Created by Zero on 26/04/2022.
//

#pragma once

#include <span>
#include <atomic>
#include <concepts>
#include <type_traits>
#include <string_view>
#include "stl.h"

namespace katana::concepts {

struct Noncopyable {
    Noncopyable() noexcept = default;
    Noncopyable(const Noncopyable &) noexcept = delete;
    Noncopyable &operator=(const Noncopyable &) noexcept = delete;
    Noncopyable(Noncopyable &&) noexcept = default;
    Noncopyable &operator=(Noncopyable &&) noexcept = default;
};

template<typename T>
concept iterable = requires(T v) {
    v.begin();
    v.end();
};

template<typename T>
concept string_viewable = requires(T v) {
    katana::string_view{v};
};

template<typename T>
concept span_convertible = requires(T v) {
    katana::span{v};
};

template<typename T, typename... Args>
concept constructible = requires(Args... args) {
    T{args...};
};

template<typename Src, typename Dest>
concept static_convertible = requires(Src s) {
    static_cast<Dest>(s);
};

template<typename Src, typename Dest>
concept bitwise_convertible = sizeof(Src) >= sizeof(Dest);

template<typename Src, typename Dest>
concept reinterpret_convertible = requires(Src s) {
    reinterpret_cast<Dest *>(&s);
};

template<typename F, typename... Args>
concept invocable = std::is_invocable_v<F, Args...>;

template<typename Ret, typename F, typename... Args>
concept invocable_with_return = std::is_invocable_r_v<Ret, F, Args...>;

template<typename T>
concept pointer = std::is_pointer_v<T>;

template<typename T>
concept non_pointer = std::negation_v<std::is_pointer<T>>;

template<typename T>
concept container = requires(T a) {
    a.begin();
    a.size();
};

namespace detail {
    template<typename... T>
    struct all_same_impl : std::true_type {};

    template<typename First, typename... Other>
    struct all_same_impl<First, Other...> : std::conjunction<std::is_same<First, Other>...> {};
}// namespace detail

template<typename... T>
using is_same = detail::all_same_impl<T...>;

template<typename... T>
constexpr auto is_same_v = is_same<T...>::value;

template<typename... T>
concept same = is_same_v<T...>;

template<typename A, typename B>
concept different = !same<A, B>;

}// namespace katana::concepts