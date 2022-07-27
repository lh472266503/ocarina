//
// Created by Zero on 15/05/2022.
//

#pragma once

#include "basic_traits.h"
#include "concepts.h"
#include "math/base.h"

namespace ocarina {

template<typename T, typename U>
requires is_integral_v<T> && is_integral_v<U>
[[nodiscard]] static constexpr auto mem_offset(T offset, U alignment) noexcept {
    return (offset + alignment - 1u) / alignment * alignment;
}


inline namespace size_literals {
[[nodiscard]] constexpr auto operator""_kb(size_t bytes) noexcept {
    return static_cast<size_t>(bytes * 1024u);
}

[[nodiscard]] constexpr auto operator""_mb(size_t bytes) noexcept {
    return static_cast<size_t>(bytes * sqr(1024u));
}

[[nodiscard]] constexpr auto operator""_gb(size_t bytes) noexcept {
    return static_cast<size_t>(bytes * Pow<3>(1024u));
}
}// namespace size_literals

class Clock {
public:
    using SystemClock = std::chrono::high_resolution_clock;
    using Tick = std::chrono::high_resolution_clock::time_point;

private:
    Tick _last;

public:
    Clock() noexcept : _last{SystemClock::now()} {}
    void start() noexcept { _last = SystemClock::now(); }
    [[nodiscard]] auto elapse_ms() const noexcept {
        auto curr = SystemClock::now();
        using namespace std::chrono_literals;
        return (curr - _last) / 1ns * 1e-6;
    }
    [[nodiscard]] auto elapse_s() const noexcept {
        return elapse_ms() * 1000;
    }
};

}// namespace ocarina