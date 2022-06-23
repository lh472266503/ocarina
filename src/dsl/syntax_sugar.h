//
// Created by Zero on 20/06/2022.
//

#pragma once

#define $if(...) detail::IfStmtBuilder::create(__VA_ARGS__) / [&]() noexcept
#define $else % [&]() noexcept
#define $elif(...) *(__VA_ARGS__) / [&]() noexcept

#define $comment(...) comment(#__VA_ARGS__)