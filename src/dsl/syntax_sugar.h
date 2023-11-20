//
// Created by Zero on 20/06/2022.
//

#pragma once

#include "syntax.h"

#define $source_location ocarina::format("{},{}", __FILE__, __LINE__)

#define $sign_source_location comment($source_location);

#define $if(...) ::ocarina::detail::IfStmtBuilder::create_with_source_location($source_location, __VA_ARGS__) / [&]() noexcept
#define $else % [&]() noexcept
#define $elif(...) *[&] {                                                                                \
    return ::ocarina::detail::IfStmtBuilder::create_with_source_location($source_location, __VA_ARGS__); \
} / [&]

#define $comment(...) ::ocarina::comment(#__VA_ARGS__);

#define $switch(...) ::ocarina::detail::SwitchStmtBuilder::create_with_source_location($source_location, __VA_ARGS__) *[&]() noexcept
#define $case(...) ::ocarina::detail::CaseStmtBuilder::create_with_source_location($source_location, __VA_ARGS__) *[&]() noexcept
#define $break ::ocarina::break_($source_location)
#define $default ::ocarina::detail::DefaultStmtBuilder($source_location) *[&]() noexcept
#define $continue ::ocarina::continue_($source_location)

#define $loop ::ocarina::detail::LoopStmtBuilder::create() *[&]() noexcept
#define $while(...) ::ocarina::detail::LoopStmtBuilder::create() / [&]() noexcept { \
    if_(!(__VA_ARGS__), [&] {                                                       \
        break_();                                                                   \
    });                                                                             \
} *[&]() noexcept

#define $for(v, ...) ::ocarina::detail::range_with_source_location($source_location, __VA_ARGS__) / [&](auto v) noexcept

#define $return(...) ::ocarina::return_(__VA_ARGS__)