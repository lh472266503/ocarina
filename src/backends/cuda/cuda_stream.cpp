//
// Created by Zero on 09/07/2022.
//

#include "cuda_stream.h"
#include "cuda_command_visitor.h"
#include "util.h"

namespace ocarina {

CUDAStream::CUDAStream() noexcept {
    OC_CU_CHECK(cuStreamCreate(&_stream, CU_STREAM_NON_BLOCKING));
    OC_CU_CHECK(cuEventCreate(&_event, CU_EVENT_DISABLE_TIMING));
}

CUDAStream::~CUDAStream() noexcept {
    OC_CU_CHECK(cuStreamDestroy(_stream));
    OC_CU_CHECK(cuEventDestroy(_event));
}

void CUDAStream::commit() noexcept {

}

}// namespace ocarina