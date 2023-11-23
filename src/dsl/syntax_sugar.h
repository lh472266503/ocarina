//
// Created by Zero on 20/06/2022.
//

#pragma once

#include "stmt_builder.h"

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

#define $loop ::ocarina::detail::LoopStmtBuilder::create_with_source_location($source_location) *[&]() noexcept
#define $while(...) ::ocarina::detail::LoopStmtBuilder::create_with_source_location($source_location) / [&]() noexcept { \
    if_(!(__VA_ARGS__), [&] {                                                                                            \
        break_();                                                                                                        \
    });                                                                                                                  \
} *[&]() noexcept

#define $for(v, ...) ::ocarina::detail::range_with_source_location($source_location, __VA_ARGS__) / [&](auto v) noexcept

#define $return(...) ::ocarina::return_(__VA_ARGS__)

#define $scope ::ocarina::detail::ScopeStmtBuilder($source_location) + [&]() noexcept

#define $debug(...) Printer::instance().set_description("debug " + $source_location).debug(__VA_ARGS__);
#define $info(...) Printer::instance().set_description("info " + $source_location).info(__VA_ARGS__);
#define $warn(...) Printer::instance().set_description("warn " + $source_location).warn(__VA_ARGS__);
#define $err(...) Printer::instance().set_description("err " + $source_location).err(__VA_ARGS__);

#define $debug_with_location(...) Printer::instance().set_description("debug " + $source_location).debug_with_location(__VA_ARGS__);
#define $info_with_location(...) Printer::instance().set_description("info " + $source_location).info_with_location(__VA_ARGS__);
#define $warn_with_location(...) Printer::instance().set_description("warn " + $source_location).warn_with_location(__VA_ARGS__);
#define $err_with_location(...) Printer::instance().set_description("err " + $source_location).err_with_location(__VA_ARGS__);

#define $debugger_execute Debugger::instance().set_description("debugger_execute " + $source_location) * [&]() noexcept