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

#ifdef OC_GENERATOR_EXPORT_DLL
#define OC_GENERATOR_API __declspec(dllexport)
#else
#define OC_GENERATOR_API __declspec(dllimport)
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

#ifdef OC_UTIL_EXPORT_DLL
#define OC_UTIL_API __declspec(dllexport)
#else
#define OC_UTIL_API __declspec(dllimport)
#endif

#ifdef OC_MARH_EXPORT_DLL
#define OC_MATH_API __declspec(dllexport)
#else
#define OC_MATH_API __declspec(dllimport)
#endif

#ifdef OC_GUI_EXPORT_DLL
#define OC_GUI_API __declspec(dllexport)
#else
#define OC_GUI_API __declspec(dllimport)
#endif

#ifdef _MSC_VER
#define OC_OFFSET_OF(type, member) __builtin_offsetof(type, member)
#else
#define OC_OFFSET_OF(type, member) offsetof(type, member)
#endif

#ifdef __cplusplus
#define OC_EXTERN_C extern "C"
#define OC_NOEXCEPT noexcept
#else
#define OC_EXTERN_C
#define OC_NOEXCEPT
#endif

#ifdef _MSC_VER
#define OC_FORCE_INLINE inline
#define OC_NEVER_INLINE __declspec(noinline)
#define OC_DLL
#define OC_EXPORT_API OC_EXTERN_C __declspec(dllexport)
#define OC_IMPORT_API OC_EXTERN_C __declspec(dllimport)
#else
#define OC_FORCE_INLINE [[gnu::always_inline, gnu::hot]] inline
#define OC_NEVER_INLINE [[gnu::noinline]]
#define OC_DLL
#define OC_EXPORT_API OC_EXTERN_C [[gnu::visibility("default")]]
#define OC_IMPORT_API OC_EXTERN_C
#endif

#define OC_USING_SV using namespace std::string_view_literals;

#define OC_ASSERT(...) assert(__VA_ARGS__)

namespace ocarina {
using handle_ty = uint64_t;
}

#define OC_NDSC_INLINE [[nodiscard]] inline
#define OC_NODISCARD [[nodiscard]]

#define CUDA_ARGUMENT_PUSH 1

#define TYPE_PREFIX "oc_"
