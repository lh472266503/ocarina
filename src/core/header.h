//
// Created by Zero on 24/04/2022.
//


#pragma once

#include <filesystem>
#include <stdint.h>
#include <cstdint>
#include <cstddef>
#include <tuple>
#include <type_traits>

#define KTN_NODISCARD [[nodiscard]]

#ifdef KTN_AST_EXPORT_DLL
#define KTN_AST_API __declspec(dllexport)
#else
#define KTN_AST_API __declspec(dllimport)
#endif

#ifdef KTN_CORE_EXPORT_DLL
#define KTN_CORE_API __declspec(dllexport)
#else
#define KTN_CORE_API __declspec(dllimport)
#endif

#ifdef KTN_DSL_EXPORT_DLL
#define KTN_DSL_API __declspec(dllexport)
#else
#define KTN_DSL_API __declspec(dllimport)
#endif