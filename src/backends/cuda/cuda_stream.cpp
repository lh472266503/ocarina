//
// Created by Zero on 09/07/2022.
//

#include "cuda_stream.h"
#include "cuda_command_visitor.h"
#include "util.h"
#include "cuda_device.h"

namespace ocarina {

CUDAStream::CUDAStream(CUDADevice *device) noexcept
    : _device(device) {
    OC_CU_CHECK(cuStreamCreate(&_stream, CU_STREAM_NON_BLOCKING));
    OC_CU_CHECK(cuEventCreate(&_event, CU_EVENT_DISABLE_TIMING));
}

CUDAStream::~CUDAStream() noexcept {
    OC_CU_CHECK(cuStreamDestroy(_stream));
    OC_CU_CHECK(cuEventDestroy(_event));
}

void CUDAStream::commit(const Commit &commit) noexcept {
    CUDACommandVisitor cmd_visitor{_device, _stream};
    for (auto &cmd : _command_queue) {
        cmd->accept(cmd_visitor);
    }
    _command_queue.clear();
}

void CUDAStream::barrier() noexcept {
    constexpr CUevent_wait_flags_enum flags = CU_EVENT_WAIT_DEFAULT;
    OC_CU_CHECK(cuEventRecord(_event, _stream));
    OC_CU_CHECK(cuStreamWaitEvent(_stream, _event, flags));
}
}// namespace ocarina