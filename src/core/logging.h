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

namespace nano {
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
}// namespace nano::core

#define NN_SOURCE_LOCATION __FILE__, ":", __LINE__

#define SET_LOG_LEVEL(lv) \
    ::nano::core::set_log_level(spdlog::level::level_enum::lv);

#define NN_DEBUG(...) \
    ::nano::core::debug(__VA_ARGS__);
#define NN_DEBUG_FORMAT(FMT, ...) \
    NN_DEBUG(nano::format(FMT, __VA_ARGS__));

#define NN_INFO(...) \
    ::nano::core::info(__VA_ARGS__);
#define NN_INFO_FORMAT(FMT, ...) \
    NN_INFO(nano::format(FMT, __VA_ARGS__));

#define NN_WARNING(...) \
    ::nano::core::warning(__VA_ARGS__, "\n    Source: ", NN_SOURCE_LOCATION);
#define NN_WARNING_IF(...) \
    ::nano::core::warning_if(__VA_ARGS__, "\n    Source: ", NN_SOURCE_LOCATION);
#define NN_WARNING_IF_NOT(...) \
    ::nano::core::warning_if_not(__VA_ARGS__, "\n    Source: ", NN_SOURCE_LOCATION);
#define NN_WARNING_FORMAT(FMT, ...) \
    NN_WARNING(nano::format(FMT, __VA_ARGS__));

#define NN_EXCEPTION(...) \
    ::nano::core::exception(__VA_ARGS__, "\n    Source: ", NN_SOURCE_LOCATION);
#define NN_EXCEPTION_IF(...) \
    ::nano::core::exception_if(__VA_ARGS__, "\n    Source: ", NN_SOURCE_LOCATION);
#define NN_EXCEPTION_IF_NOT(...) \
    ::nano::core::exception_if_not(__VA_ARGS__, "\n    Source: ", NN_SOURCE_LOCATION);
#define NN_EXCEPTION_FORMAT(FMT, ...) \
    NN_EXCEPTION(nano::format(FMT, __VA_ARGS__));

#define NN_ERROR(...) \
    ::nano::core::error(__VA_ARGS__, "\n    Source: ", NN_SOURCE_LOCATION);
#define NN_ERROR_IF(...) \
    ::nano::core::error_if(__VA_ARGS__, "\n    Source: ", NN_SOURCE_LOCATION);
#define NN_ERROR_IF_NOT(...) \
    ::nano::core::error_if_not(__VA_ARGS__, "\n    Source: ", NN_SOURCE_LOCATION);
#define NN_ERROR_FORMAT(FMT, ...) \
    NN_ERROR(nano::format(FMT, __VA_ARGS__));