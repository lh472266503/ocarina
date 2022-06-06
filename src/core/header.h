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

#ifdef NN_AST_EXPORT_DLL
#define NN_AST_API __declspec(dllexport)
#else
#define NN_AST_API __declspec(dllimport)
#endif

#ifdef NN_CORE_EXPORT_DLL
#define NN_CORE_API __declspec(dllexport)
#else
#define NN_CORE_API __declspec(dllimport)
#endif

#ifdef NN_DSL_EXPORT_DLL
#define NN_DSL_API __declspec(dllexport)
#else
#define NN_DSL_API __declspec(dllimport)
#endif

#ifdef NN_COMPILE_EXPORT_DLL
#define NN_COMPILE_API __declspec(dllexport)
#else
#define NN_COMPILE_API __declspec(dllimport)
#endif

#ifdef NN_RUNTIME_EXPORT_DLL
#define NN_RUNTIME_API __declspec(dllexport)
#else
#define NN_RUNTIME_API __declspec(dllimport)
#endif

#ifdef NN_BACKENDS_EXPORT_DLL
#define NN_BACKENDS_API __declspec(dllexport)
#else
#define NN_BACKENDS_API __declspec(dllimport)
#endif

#ifdef _MSC_VER
#define NN_OFFSET_OF(type, member) __builtin_offsetof(type, member)
#else
#define NN_OFFSET_OF(type, member) offsetof(type, member)
#endif

#define NN_USING_SV using namespace std::string_view_literals;

#define NN_ASSERT(...) assert(__VA_ARGS__)

