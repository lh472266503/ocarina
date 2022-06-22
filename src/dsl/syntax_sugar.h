//
// Created by Zero on 20/06/2022.
//

#pragma once

#define $if(...) IfStmtBuilder::create(__VA_ARGS__) / [&]() noexcept
#define $else % [&]() noexcept
#define $elif(...) IfStmtBuilder::create(__VA_ARGS__) / [&]() noexcept