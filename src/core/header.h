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

#ifdef _MSC_VER
#define KTN_OFFSET_OF(type, member) __builtin_offsetof(type, member)
#else
#define KTN_OFFSET_OF(type, member) offsetof(type, member)
#endif

#define KTN_REQUIRES(args) std::enable_if_t<args, int> = 0