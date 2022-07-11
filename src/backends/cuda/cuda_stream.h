//
// Created by Zero on 09/07/2022.
//

#pragma once

#include "core/stl.h"
#include "runtime/stream.h"
#include <cuda.h>

namespace ocarina {
class CUDAStream : public Stream::Impl {
private:
    CUstream _stream{};
    CUevent _event{};

public:
    CUDAStream() noexcept;

    ~CUDAStream() noexcept;

    void add_command(Command *cmd) noexcept override {
        _command_queue.push_back(cmd);
    }

    void synchronize() noexcept override;
    void barrier() noexcept override;
    void commit() noexcept override;
};
}// namespace ocarina