//
// Created by Zero on 25/12/2023.
//

#pragma once

#include "printer.h"
#include "debugger.h"

namespace ocarina {

class Env {
private:
    Env() = default;
    Env(const Env &) = delete;
    Env(Env &&) = delete;
    Env operator=(const Env &) = delete;
    Env operator=(Env &&) = delete;
    static Env *s_env;

private:
    Printer _printer;
    Debugger _debugger;

    /// Check if the array or buffer is over boundary
    bool _valid_check{true};
    bool _code_obfuscation{false};

public:
    [[nodiscard]] static Env &instance() noexcept;
    static void destroy_instance() noexcept;
    [[nodiscard]] static Printer &printer() noexcept { return instance()._printer; }
    [[nodiscard]] static Debugger &debugger() noexcept { return instance()._debugger; }
    void init(Device &device) {
        _printer.init(device);
        _debugger.init(device);
    }
    [[nodiscard]] static bool is_printer_enabled() noexcept { return printer().enabled(); }
    template<typename T>
    requires is_scalar_expr_v<T> || is_vector_expr_v<T>
    [[nodiscard]] T zero_if_nan_inf(T &&value) const noexcept {
        if (!valid_check()) {
            return OC_FORWARD(value);
        }
        string tb = traceback_string(-1);
        if constexpr (is_scalar_expr_v<T>) {
            string content = ocarina::format("traceback is {}", tb.c_str());
            if_(ocarina::isnan(value) || ocarina::isinf(value), [&] {
                $err_with_location("invalid value : {} \n" + content, value);
            });
        } else if constexpr (is_vector2_expr_v<T>) {
            string content = ocarina::format("traceback is {}", tb.c_str());
            if_(ocarina::has_nan(value) || ocarina::has_inf(value), [&] {
                $err_with_location("invalid value : ({}, {}) \n" + content, value);
            });
        } else if constexpr (is_vector3_expr_v<T>) {
            string content = ocarina::format("traceback is {}", tb.c_str());
            if_(ocarina::has_nan(value) || ocarina::has_inf(value), [&] {
                $err_with_location("invalid value : ({}, {}, {}) \n" + content, value);
            });
        } else if constexpr (is_vector4_expr_v<T>) {
            string content = ocarina::format("traceback is {}", tb.c_str());
            if_(ocarina::has_nan(value) || ocarina::has_inf(value), [&] {
                $err_with_location("invalid value : ({}, {}, {}, {}) \n" + content, value);
            });
        }
    }
    static void set_printer_enabled(bool enabled) noexcept { printer().set_enabled(enabled); }
    [[nodiscard]] static bool valid_check() noexcept { return instance()._valid_check; }
    static void set_valid_check(bool val) noexcept { instance()._valid_check = val; }
    [[nodiscard]] static bool code_obfuscation() noexcept { return instance()._code_obfuscation; }
    static void set_code_obfuscation(bool val) noexcept { instance()._code_obfuscation = val; }
};

}// namespace ocarina