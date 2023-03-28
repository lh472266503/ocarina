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
#include <dsl/syntax.h>
#include <dsl/builtin.h>
#include <dsl/operators.h>

namespace ocarina {

class Printer {
public:
    struct Item {
        spdlog::level::level_enum level;
        string fmt;
        uint size;
    };

private:
    bool _has_reset{};
    Managed<uint> _buffer;
    spdlog::logger _logger;
    Device &_device;
    vector<Item> _items;

private:
    template<typename... Args>
    void _log_buffer(Uint offset,const Args &...args) noexcept {

    }

    template<typename... Args>
    void _log(spdlog::level::level_enum level, const string &fmt, const Args &...args) noexcept {
        static constexpr uint count = sizeof...(Args);
        uint last = static_cast<uint>(_buffer.device().size() - 1);
        Uint offset = _buffer.atomic(last).fetch_add(count);
        uint item_index = _items.size();
        if_(offset < last, [&] {
            _buffer.write(offset, item_index);
        });
        if_(offset + count < last, [&]{
            _log_buffer(offset + 1, OC_FORWARD(args)...);
        });
    }

public:
    explicit Printer(Device &device, size_t capacity = 16_mb)
        : _device(device),
          _logger{logger()} {
        _buffer.device() = device.create_buffer<uint>(capacity / sizeof(uint));
        _buffer.host().resize(capacity);
    }

    template<typename... Args>
    void log_debug(const string &fmt, const Args &...args) noexcept {
        _log(spdlog::level::level_enum::debug, fmt, OC_FORWARD(args)...);
    }

    template<typename... Args>
    void log_info(const string &fmt, const Args &...args) noexcept {
        _log(spdlog::level::level_enum::info, fmt, OC_FORWARD(args)...);
    }

    template<typename... Args>
    void log_warning(const string &fmt, const Args &...args) noexcept {
        _log(spdlog::level::level_enum::warn, fmt, OC_FORWARD(args)...);
    }

    template<typename... Args>
    void log_error(const string &fmt, const Args &...args) noexcept {
        _log(spdlog::level::level_enum::err, fmt, OC_FORWARD(args)...);
    }
};

}// namespace ocarina