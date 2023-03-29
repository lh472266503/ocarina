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
        std::function<void(const uint *)> func;
        uint size;
    };

private:
    bool _has_reset{};
    Managed<uint> _buffer;
    spdlog::logger _logger;
    Device &_device;
    vector<Item> _items;

private:
    void _log_to_buffer(Uint offset, uint index) noexcept {}

    template<typename Current, typename... Args>
    void _log_to_buffer(Uint offset, uint index, const Current &cur, const Args &...args) noexcept {
        using type = expr_value_t<Current>;
        if constexpr (is_integral_v<type> || is_boolean_v<type>) {
            _buffer.write(offset + index, cast<uint>(cur));
        } else if constexpr (is_floating_point_v<type>) {
            _buffer.write(offset + index, as<uint>(cur));
        } else {
            static_assert(always_false_v<type>, "unsupported type for printing in kernel.");
        }
        _log_to_buffer(offset, index + 1, args...);
    }

    template<typename... Args>
    void _log(spdlog::level::level_enum level, const string &fmt, const Args &...args) noexcept {
        static constexpr uint count = sizeof...(Args);
        uint last = static_cast<uint>(_buffer.device().size() - 1);
        Uint offset = _buffer.atomic(last).fetch_add(count + 1);
        uint item_index = _items.size();
        if_(offset < last, [&] {
            _buffer.write(offset, item_index);
        });
        if_(offset + count < last, [&] {
            _log_to_buffer(offset + 1, 0, OC_FORWARD(args)...);
        });
        auto decode = [this, level, fmt, tuple_args = std::tuple<expr_value_t<Args>...>()](const uint *data) -> void {
            auto decode_arg = [tuple_args, data]<size_t i>() noexcept {
                using Arg = expr_value_t<std::tuple_element_t<i, std::tuple<Args...>>>;
                if constexpr (is_integral_v<Arg> || is_boolean_v<Arg>) {
                    return static_cast<Arg>(data[i]);
                } else {
                    return bit_cast<Arg>(data[i]);
                }
            };
            auto host_print = [&]<size_t ...i>(std::index_sequence<i...>) {
                _logger.log(level, fmt, decode_arg.template operator()<i>()...);
            };
            host_print(std::index_sequence_for<Args...>());
        };
        _items.push_back({decode, count});
    }

public:
    explicit Printer(Device &device, size_t capacity = 16_mb)
        : _device(device),
          _logger{logger()} {
        capacity /= sizeof(uint);
        _buffer.device() = device.create_buffer<uint>(capacity);
        _buffer.host().reserve(capacity);
        reset();
    }

    void reset() {
        _buffer.device().clear_immediately();
        _buffer.host().resize(_buffer.capacity(), 0);
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

    void retrieve_immediately() noexcept;
};

}// namespace ocarina