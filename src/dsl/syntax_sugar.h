//
// Created by Zero on 20/06/2022.
//

#pragma once

#include "syntax.h"

#define $if(...) ::ocarina::detail::IfStmtBuilder::create(__VA_ARGS__) / [&]() noexcept
#define $else % [&]() noexcept
#define $elif(...) *(__VA_ARGS__) / [&]() noexcept

#define $comment(...) ::ocarina::comment(#__VA_ARGS__);

#define $switch(...) ::ocarina::detail::SwitchStmtBuilder::create(__VA_ARGS__) *[&]() noexcept
#define $case(...) ::ocarina::detail::CaseStmtBuilder::create(__VA_ARGS__) *[&]() noexcept
#define $break ::ocarina::break_()
#define $default ::ocarina::detail::DefaultStmtBuilder() *[&]() noexcept
#define $continue ::ocarina::continue_()

#define $loop ::ocarina::detail::LoopStmtBuilder::create() *[&]() noexcept
#define $while(cond) ::ocarina::detail::LoopStmtBuilder::create() / [&]() noexcept { \
    if_(!cond, [&] {                                                                 \
        break_();                                                                    \
    });                                                                              \
} *[&]() noexcept
