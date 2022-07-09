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

    void synchronize() noexcept override {
    }

    void barrier() noexcept override {
    }

    void flush() noexcept override {
    }
};
}// namespace ocarina