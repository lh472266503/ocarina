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

namespace detail {
template<typename Current>
auto args_to_tuple(const Current &cur) {
    if constexpr (is_scalar_expr_v<Current>) {
        return std::tuple{cur};
    } else if constexpr (is_vector2_expr_v<Current>) {
        return std::tuple{cur[0], cur[1]};
    } else if constexpr (is_vector3_expr_v<Current>) {
        return std::tuple{cur[0], cur[1], cur[2]};
    } else if constexpr (is_vector4_expr_v<Current>) {
        return std::tuple{cur[0], cur[1], cur[2], cur[3]};
    } else {
        static_assert(always_false_v<Current>);
    }
}

template<typename Current, typename... Args>
auto args_to_tuple(const Current &cur, const Args &...args) {
    return std::tuple_cat(args_to_tuple(cur), args_to_tuple(args...));
}
}// namespace detail

class Printer {
public:
    struct Item {
        std::function<void(const uint *)> func;
        uint size;
    };

private:
    Managed<uint> _buffer;
    spdlog::logger _logger{logger()};
    vector<Item> _items;

private:
    void _log_to_buffer(Uint offset, uint index) noexcept {}

    template<typename Current, typename... Args>
    void _log_to_buffer(Uint offset, uint index, const Current &cur, const Args &...args) noexcept {
        using type = expr_value_t<Current>;
        if constexpr (is_dsl_v<Current>) {
            if constexpr (is_integral_v<type> || is_boolean_v<type>) {
                _buffer.write(offset + index, cast<uint>(cur));
            } else if constexpr (is_floating_point_v<type>) {
                _buffer.write(offset + index, as<uint>(cur));
            } else {
                static_assert(always_false_v<type>, "unsupported type for printing in kernel.");
            }
            index ++;
        }
        _log_to_buffer(offset, index, args...);
    }

    template<typename ...Args>
    void _logs(spdlog::level::level_enum level, const string &fmt, const Args &...args) noexcept {
        auto args_tuple = detail::args_to_tuple(args...);
        auto func = [&]<size_t ...i>(std::index_sequence<i...> a) {
            return _log(level, fmt, std::get<i>(args_tuple)...);
        };
        func(std::make_index_sequence<std::tuple_size_v<decltype(args_tuple)>>());
    }

    template<typename... Args>
    void _log(spdlog::level::level_enum level, const string &fmt, const Args &...args) noexcept {
        constexpr auto count = (0u + ... + static_cast<uint>(is_dsl_v<Args>));
        uint last = static_cast<uint>(_buffer.device().size() - 1);
        Uint offset = _buffer.atomic(last).fetch_add(count + 1);
        uint item_index = _items.size();
        if_(offset < last, [&] {
            _buffer.write(offset, item_index);
        });
        if_(offset + count < last, [&] {
            _log_to_buffer(offset + 1, 0, OC_FORWARD(args)...);
        });

        uint dsl_counter = 0;
        // todo change to index_sequence
        auto convert = [&](const auto &arg) noexcept {
            using T = std::remove_cvref_t<decltype(arg)>;
            if constexpr (is_dsl_v<T>) {
                return dsl_counter++;
            } else {
                return arg;
            }
        };

        auto decode = [this, level, fmt, tuple_args = std::tuple{convert(args)...}](const uint *data) -> void {
            auto decode_arg = [tuple_args, data]<size_t i>() noexcept {
                using Arg = std::tuple_element_t<i, std::tuple<Args...>>;
                if constexpr (is_dsl_v<Arg>) {
                    if constexpr (is_integral_v<Arg> || is_boolean_v<Arg>) {
                        return static_cast<expr_value_t<Arg>>(data[std::get<i>(tuple_args)]);
                    } else {
                        return bit_cast<expr_value_t<Arg>>(data[std::get<i>(tuple_args)]);
                    }
                } else {
                    return std::get<i>(tuple_args);
                }
            };
            auto host_print = [&]<size_t... i>(std::index_sequence<i...>) {
                _logger.log(level, fmt, decode_arg.template operator()<i>()...);
            };
            host_print(std::index_sequence_for<Args...>());
        };
        _items.push_back({decode, count});
    }

private:
    Printer() = default;
    Printer(const Printer &) = delete;
    Printer(Printer &&) = delete;
    Printer operator=(const Printer &) = delete;
    Printer operator=(Printer &&) = delete;
    static Printer * s_printer;

public:
    [[nodiscard]] static Printer &instance() noexcept;
    [[nodiscard]] static void destroy_instance() noexcept;

    void init(Device &device, size_t capacity = 16_mb) {
        capacity /= sizeof(uint);
        _buffer.device() = device.create_buffer<uint>(capacity);
        _buffer.host().reserve(capacity);
        reset();
    }

    void reset() {
        _buffer.device().clear_immediately();
        _buffer.resize(_buffer.capacity());
    }

    template<typename... Args>
    void log_debug(const string &fmt, const Args &...args) noexcept {
        _logs(spdlog::level::level_enum::debug, fmt, OC_FORWARD(args)...);
    }

    template<typename... Args>
    void log_info(const string &fmt, const Args &...args) noexcept {
        _logs(spdlog::level::level_enum::info, fmt, OC_FORWARD(args)...);
    }

    template<typename... Args>
    void log_warning(const string &fmt, const Args &...args) noexcept {
        _logs(spdlog::level::level_enum::warn, fmt, OC_FORWARD(args)...);
    }

    template<typename... Args>
    void log_error(const string &fmt, const Args &...args) noexcept {
        _logs(spdlog::level::level_enum::err, fmt, OC_FORWARD(args)...);
    }

    void retrieve_immediately() noexcept;
};

}// namespace ocarina