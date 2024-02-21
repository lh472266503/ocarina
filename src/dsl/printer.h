//
// Created by Zero on 28/03/2023.
//

#pragma once

#include <mutex>
#include <thread>
#include <ast/function.h>
#include "rhi/resources/managed.h"
#include <dsl/computable.h>
#include <dsl/var.h>
#include <dsl/stmt_builder.h>
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
    using OutputFunc = std::function<void(int, const char *)>;

    struct Item {
        ocarina::move_only_function<void(const uint *, const OutputFunc &)> func;
        uint size;
    };

private:
    Managed<uint> _buffer;
    spdlog::logger _logger{logger()};
    vector<Item> _items;
    bool _enabled{true};
    mutable string _desc;

private:
    void _log_to_buffer(Uint offset, uint index) noexcept {}
    template<typename Current, typename... Args>
    void _log_to_buffer(Uint offset, uint index, const Current &cur, const Args &...args) noexcept;
    template<typename... Args>
    void _logs(spdlog::level::level_enum level, const string &fmt, const Args &...args) noexcept;
    template<typename... Args>
    void _log(spdlog::level::level_enum level, const string &fmt, const Args &...args) noexcept;

private:
    Printer() = default;
    Printer(const Printer &) = delete;
    Printer(Printer &&) = delete;
    Printer operator=(const Printer &) = delete;
    Printer operator=(Printer &&) = delete;
    friend class Env;

public:
    void init(Device &device, size_t capacity = 16_mb) {
        capacity /= sizeof(uint);
        _buffer.reset_all(device, capacity, "Printer::_buffer");
        reset();
    }

    OC_MAKE_MEMBER_GETTER_SETTER(enabled, )

    Printer &set_description(const string &desc) noexcept {
        _desc = desc;
        return *this;
    }
    [[nodiscard]] Managed<uint> &buffer() noexcept { return _buffer; }
    [[nodiscard]] const Managed<uint> &buffer() const noexcept { return _buffer; }
    [[nodiscard]] uint element_num() const noexcept { return _buffer.host_buffer().back(); }

    void reset() {
        _buffer.device_buffer().reset_immediately();
        _buffer.resize(_buffer.capacity());
    }

#define OC_MAKE_LOG_FUNC(level_name)                                                                          \
    template<typename... Args>                                                                                \
    void level_name(const string &fmt, const Args &...args) noexcept {                                        \
        _logs(spdlog::level::level_enum::level_name, fmt, OC_FORWARD(args)...);                               \
    }                                                                                                         \
    template<typename... Args>                                                                                \
    void level_name##_with_location(const string &fmt, const Args &...args) noexcept {                        \
        Uint3 idx = dispatch_idx();                                                                           \
        Uint id = dispatch_id();                                                                              \
        level_name(fmt + " [with thread idx({}, {}, {}), id({})]", OC_FORWARD(args)..., idx, id);             \
    }                                                                                                         \
    template<typename... Args>                                                                                \
    void level_name##_with_traceback(const string &fmt, const Args &...args) noexcept {                       \
        string tb = ocarina::format(" traceback is :\n {}", traceback_string(1));                             \
        level_name##_with_location(fmt + tb, OC_FORWARD(args)...);                                            \
    }                                                                                                         \
    template<typename... Args>                                                                                \
    void level_name##_if(const Bool &cond, const string &fmt, const Args &...args) noexcept {                 \
        if_(cond, [&] {                                                                                       \
            level_name(fmt, OC_FORWARD(args)...);                                                             \
        });                                                                                                   \
    }                                                                                                         \
    template<typename... Args>                                                                                \
    void level_name##_with_location##_if(const Bool &cond, const string &fmt, const Args &...args) noexcept { \
        if_(cond, [&] {                                                                                       \
            level_name##_with_location(fmt, OC_FORWARD(args)...);                                             \
        });                                                                                                   \
    }                                                                                                         \
    template<typename... Args>                                                                                \
    void level_name##_with_traceback_if(const Bool &cond, const string &fmt, const Args &...args) noexcept {  \
        string tb = ocarina::format(" traceback is :\n {}", traceback_string(1));                             \
        level_name##_with_location##_if(cond, fmt + tb, OC_FORWARD(args)...);                                 \
    }

    OC_MAKE_LOG_FUNC(debug)
    OC_MAKE_LOG_FUNC(info)
    OC_MAKE_LOG_FUNC(warn)
    OC_MAKE_LOG_FUNC(err)

#undef OC_MAKE_LOG_FUNC
    void retrieve_immediately(const OutputFunc &func = nullptr) noexcept;
    [[nodiscard]] CommandList retrieve(const OutputFunc &func = nullptr) noexcept;
    void output_log(const OutputFunc &func = nullptr) noexcept;
};

template<typename Current, typename... Args>
void Printer::_log_to_buffer(Uint offset, uint index, const Current &cur, const Args &...args) noexcept {
    using type = expr_value_t<Current>;
    if constexpr (is_dsl_v<Current>) {
        if constexpr (is_integral_v<type> || is_boolean_v<type>) {
            _buffer.write(offset + index, cast<uint>(cur), false);
        } else if constexpr (is_floating_point_v<type>) {
            _buffer.write(offset + index, as<uint>(cur), false);
        } else {
            static_assert(always_false_v<type>, "unsupported type for printing in kernel.");
        }
        index++;
    }
    _log_to_buffer(offset, index, args...);
}

template<typename... Args>
void Printer::_logs(spdlog::level::level_enum level, const string &fmt, const Args &...args) noexcept {
    auto args_tuple = detail::args_to_tuple(args...);
    auto func = [&]<size_t... i>(std::index_sequence<i...> a) {
        _log(level, fmt, std::get<i>(args_tuple)...);
    };
    func(std::make_index_sequence<std::tuple_size_v<decltype(args_tuple)>>());
}

template<typename... Args>
void Printer::_log(spdlog::level::level_enum level, const string &fmt, const Args &...args) noexcept {
    if (!_enabled) {
        return;
    }

    if (!_desc.empty()) {
        comment(_desc);
    }

    constexpr array<uint, sizeof...(Args)> size_arr = {(is_dsl_v<Args> ? (sizeof(expr_value_t<Args>) / sizeof(uint)) : 0u)...};

    static constexpr uint total_size = std::accumulate(size_arr.begin(), size_arr.end(), 0);

    uint last = static_cast<uint>(_buffer.device_buffer().size() - 1);
    Uint item_index = static_cast<uint>(_items.size());

    comment("start log >>>>>>>>>> ");
    outline([&] {
        Uint offset = _buffer.atomic(last).fetch_add(total_size + 1);
        if_(offset < last, [&] {
            _buffer.write(offset, item_index, false);
        });
        if_(offset + total_size < last, [&] {
            _log_to_buffer(offset + 1, 0, OC_FORWARD(args)...);
        });
    },
            "log output");
    comment("end log <<<<<<<<<<");
    _desc = "";

    uint offset = 0;
    uint cursor = 0;
    // todo change to index_sequence
    auto convert = [&](const auto &arg) noexcept {
        using T = std::remove_cvref_t<decltype(arg)>;
        uint old_cursor = cursor ++;
        if constexpr (is_dsl_v<T>) {
            uint old_offset = offset;
            offset += size_arr[old_cursor];
            return old_offset;
        } else {
            return arg;
        }
    };
    auto tuple_arg = std::tuple{convert(args)...};

    auto decode = [this, level, fmt, tuple_args = std::tuple{convert(args)...}](const uint *data, const OutputFunc &func) {
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
        auto format = [&]<size_t... i>(std::index_sequence<i...>) -> string {
            return ocarina::format(fmt, decode_arg.template operator()<i>()...);
        };
        string content = format(std::index_sequence_for<Args...>());
        if (func) {
            func(to_underlying(level), content.c_str());
        } else {
            _logger.log(level, content);
        }
    };
    _items.push_back({decode, total_size});
}

}// namespace ocarina