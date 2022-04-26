//
// Created by Zero on 26/04/2022.
//

#pragma once

#include <span>
#include <atomic>
#include <concepts>
#include <type_traits>
#include <string_view>

namespace sycamore {
inline namespace concepts {

struct Noncopyable {
    Noncopyable() noexcept = default;
    Noncopyable(const Noncopyable &) noexcept = delete;
    Noncopyable &operator=(const Noncopyable &) noexcept = delete;
    Noncopyable(Noncopyable &&) noexcept = default;
    Noncopyable &operator=(Noncopyable &&) noexcept = default;
};



}

}