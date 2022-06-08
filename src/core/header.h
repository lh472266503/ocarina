//
// Created by Zero on 24/04/2022.
//

#pragma once

#include <filesystem>
#include <cstdint>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <type_traits>

#ifdef OC_AST_EXPORT_DLL
#define OC_AST_API __declspec(dllexport)
#else
#define OC_AST_API __declspec(dllimport)
#endif

#ifdef OC_CORE_EXPORT_DLL
#define OC_CORE_API __declspec(dllexport)
#else
#define OC_CORE_API __declspec(dllimport)
#endif

#ifdef OC_DSL_EXPORT_DLL
#define OC_DSL_API __declspec(dllexport)
#else
#define OC_DSL_API __declspec(dllimport)
#endif

#ifdef OC_COMPILE_EXPORT_DLL
#define OC_COMPILE_API __declspec(dllexport)
#else
#define OC_COMPILE_API __declspec(dllimport)
#endif

#ifdef OC_RUNTIME_EXPORT_DLL
#define OC_RUNTIME_API __declspec(dllexport)
#else
#define OC_RUNTIME_API __declspec(dllimport)
#endif

#ifdef OC_BACKENDS_EXPORT_DLL
#define OC_BACKENDS_API __declspec(dllexport)
#else
#define OC_BACKENDS_API __declspec(dllimport)
#endif

#ifdef _MSC_VER
#define OC_OFFSET_OF(type, member) __builtin_offsetof(type, member)
#else
#define OC_OFFSET_OF(type, member) offsetof(type, member)
#endif

#define OC_USING_SV using namespace std::string_view_literals;

#define OC_ASSERT(...) assert(__VA_ARGS__)

