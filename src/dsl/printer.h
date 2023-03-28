//
// Created by Zero on 28/03/2023.
//

#pragma once

#include <mutex>
#include <thread>
#include <ast/function.h>
#include <rhi/managed.h>
#include <dsl/computable.h>
#include <dsl/var.h>
#include <dsl/builtin.h>
#include <dsl/operators.h>

namespace ocarina {

class Printer {
private:
    Managed<uint> _buffer;
    spdlog::logger _logger;
    Device &_device;

public:
    explicit Printer(Device &device, size_t capacity = 16_mb)
        : _device(device),
          _logger{logger()} {
        _buffer.device() = device.create_buffer<uint>(capacity);
        _buffer.host().resize(capacity);
    }
};

}// namespace ocarina