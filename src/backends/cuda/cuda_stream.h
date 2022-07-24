//
// Created by Zero on 09/07/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/stream.h"
#include <cuda.h>

namespace ocarina {
class CUDADevice;
class CUDAStream : public Stream::Impl {
private:
    CUstream _stream{};
    CUevent _event{};
    CUDADevice *_device{};
public:
    explicit CUDAStream(CUDADevice *device) noexcept;

    ~CUDAStream() noexcept;

    void add_command(Command *cmd) noexcept override {
        _command_queue.push_back(cmd);
    }

    void barrier() noexcept override;
    void commit(const Commit &cmt) noexcept override;
};
}// namespace ocarina