//
// Created by Zero on 20/06/2022.
//

#pragma once

#include "syntax.h"

#define $if(...) detail::IfStmtBuilder::create(__VA_ARGS__) / [&]() noexcept
#define $else % [&]() noexcept
#define $elif(...) *(__VA_ARGS__) / [&]() noexcept

#define $comment(...) comment(#__VA_ARGS__);

#define $switch(...) detail::SwitchStmtBuilder::create(__VA_ARGS__) * [&]() noexcept
#define $case(...) detail::CaseStmtBuilder::create(__VA_ARGS__) * [&]() noexcept
#define $break break_()
#define $default detail::DefaultStmtBuilder() * [&]() noexcept
#define $continue continue_()

