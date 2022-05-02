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

#define SCM_NODISCARD [[nodiscard]]

//#define SCM_AST_EXPORT_DLL

#ifdef SCM_AST_EXPORT_DLL
#define SCM_AST_API __declspec(dllexport)
#else
#define SCM_AST_API __declspec(dllimport)
#endif

#ifdef SCM_CORE_EXPORT_DLL
#define SCM_CORE_API __declspec(dllexport)
#else
#define SCM_CORE_API __declspec(dllimport)
#endif