//
// Created by Zero on 03/07/2022.
//

#pragma once
#include "core/logging.h"

#define OC_CUDA_CHECK(EXPR)                                                                                    \
    [&] {                                                                                                      \
        if ((EXPR) != cudaSuccess) {                                                                           \
            cudaError_t error = cudaGetLastError();                                                            \
            OC_ERROR_FORMAT("CUDA rhi error: {} at {}:{}", cudaGetErrorString(error), __FILE__, __LINE__); \
            std::abort();                                                                                      \
        }                                                                                                      \
    }()

#define OC_CU_CHECK(EXPR)                                                               \
    [&] {                                                                               \
        CUresult result = EXPR;                                                         \
        if (result != CUDA_SUCCESS) {                                                   \
            const char *str;                                                            \
            assert(CUDA_SUCCESS == cuGetErrorString(result, &str));                     \
            OC_ERROR_FORMAT("CUDA driver error: {} at {}:{}", str, __FILE__, __LINE__); \
            std::abort();                                                               \
        }                                                                               \
    }()

#define OC_NVRTC_CHECK(EXPR)                                                                            \
    [&] {                                                                                               \
        nvrtcResult code = EXPR;                                                                        \
        if (code != NVRTC_SUCCESS) {                                                                    \
            OC_ERROR_FORMAT("nvrtc compile error: {} at {}:{}", std::string(nvrtcGetErrorString(code)), \
                            __FILE__, __LINE__)                                                         \
            std::abort();                                                                               \
        }                                                                                               \
    }();
