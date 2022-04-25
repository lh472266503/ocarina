//
// Created by Zero on 24/04/2022.
//

#pragma once

#include "string_util.h"
#include <exception>
#include <iostream>
#include <filesystem>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace sycamore {
inline namespace core {
spdlog::logger &logger() noexcept;

inline void set_log_level(spdlog::level::level_enum lvl) noexcept {
    logger().set_level(lvl);
}

template<typename... Args>
inline void debug(Args &&...args) noexcept {
    logger().debug(serialize(std::forward<Args>(args)...));
}

template<typename... Args>
inline void info(Args &&...args) noexcept {
    logger().info(serialize(std::forward<Args>(args)...));
}

template<typename... Args>
inline void warning(Args &&...args) noexcept {
    logger().warn(serialize(std::forward<Args>(args)...));
}

template<typename... Args>
inline void warning_if(bool predicate, Args &&...args) noexcept {
    if (predicate) { warning(std::forward<Args>(args)...); }
}

template<typename... Args>
inline void warning_if_not(bool predicate, Args &&...args) noexcept {
    warning_if(!predicate, std::forward<Args>(args)...);
}

template<typename... Args>
[[noreturn]] inline void exception(Args &&...args) {
    throw std::runtime_error{serialize(std::forward<Args>(args)...)};
}

template<typename... Args>
inline void exception_if(bool predicate, Args &&...args) {
    if (predicate) { exception(std::forward<Args>(args)...); }
}

template<typename... Args>
inline void exception_if_not(bool predicate, Args &&...args) {
    exception_if(!predicate, std::forward<Args>(args)...);
}

template<typename... Args>
[[noreturn]] inline void error(Args &&...args) {
    logger().error(serialize(std::forward<Args>(args)...));
    exit(-1);
}

template<typename... Args>
inline void error_if(bool predicate, Args &&...args) {
    if (predicate) { error(std::forward<Args>(args)...); }
}

template<typename... Args>
inline void error_if_not(bool predicate, Args &&...args) {
    error_if(!predicate, std::forward<Args>(args)...);
}
}
}// namespace sycamore::core