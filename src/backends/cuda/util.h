//
// Created by Zero on 03/07/2022.
//

#pragma once


#define OC_CUDA_CHECK(EXPR)                                                                                  \
    [&] {                                                                                                    \
        if ((EXPR) != cudaSuccess) {                                                                         \
            cudaError_t error = cudaGetLastError();                                                          \
            OC_ERROR_FORMAT("CUDA runtime error: {} at {}:{}", cudaGetErrorString(error), __FILE__, __LINE__); \
            std::abort();                                                                                    \
        }                                                                                                    \
    }()

#define OC_CU_CHECK(EXPR)                                                             \
    [&] {                                                                             \
        CUresult result = EXPR;                                                       \
        if (result != CUDA_SUCCESS) {                                                 \
            const char *str;                                                          \
            assert(CUDA_SUCCESS == cuGetErrorString(result, &str));                   \
            OC_ERROR_FORMAT("CUDA driver error: {} at {}:{}", str, __FILE__, __LINE__); \
            std::abort();                                                             \
        }                                                                             \
    }()

    