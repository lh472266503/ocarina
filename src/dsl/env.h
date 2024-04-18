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
    Printer printer_;
    Debugger debugger_;
    mutable ocarina::map<string, basic_variant_var_t> global_vars_;

    /// Check if the array or buffer is over boundary
    bool valid_check_{true};
    bool code_obfuscation_{false};

public:
    [[nodiscard]] static Env &instance() noexcept;
    static void destroy_instance() noexcept;

    template<typename T>
    void set(const string &key, const T &val) noexcept {
        global_vars_.insert(make_pair(key, val));
    }

    template<typename T>
    [[nodiscard]] T &get(const string &key) const noexcept {
        return std::get<T>(global_vars_.at(key));
    }

    template<typename T, typename Func>
    void execute_if(const string &key, Func func) const noexcept {
        if (!has(key)) {
            return;
        }
        func(get<T>(key));
    }

    [[nodiscard]] bool has(const string &key) const noexcept {
        return global_vars_.contains(key);
    }

    void clear_global_vars() const noexcept {
        global_vars_.clear();
    }

    [[nodiscard]] static Printer &printer() noexcept { return instance().printer_; }
    [[nodiscard]] static Debugger &debugger() noexcept { return instance().debugger_; }
    void init(Device &device) {
        printer_.init(device);
        debugger_.init(device);
    }
    [[nodiscard]] static bool is_printer_enabled() noexcept { return printer().enabled(); }
    template<typename T>
    requires is_scalar_expr_v<T> || is_vector_expr_v<T>
    [[nodiscard]] T zero_if_nan_inf(T value) const noexcept {
        if (!valid_check()) {
            return value;
        }
        T ret = value;
        string tb = traceback_string(-1);
        if constexpr (is_scalar_expr_v<T>) {
            string content = ocarina::format("traceback is {}", tb.c_str());
            if_(ocarina::isnan(value) || ocarina::isinf(value), [&] {
                $err_with_location("invalid value : {} \n" + content, value);
                ret = 0.f;
            });
            return ret;
        } else if constexpr (is_vector2_expr_v<T>) {
            string content = ocarina::format("traceback is {}", tb.c_str());
            if_(ocarina::has_nan(value) || ocarina::has_inf(value), [&] {
                $err_with_location("invalid value : ({}, {}) \n" + content, value);
                ret = make_float2(0.f);
            });
            return ret;
        } else if constexpr (is_vector3_expr_v<T>) {
            string content = ocarina::format("traceback is {}", tb.c_str());
            if_(ocarina::has_nan(value) || ocarina::has_inf(value), [&] {
                $err_with_location("invalid value : ({}, {}, {}) \n" + content, value);
                ret = make_float3(0.f);
            });
            return ret;
        } else if constexpr (is_vector4_expr_v<T>) {
            string content = ocarina::format("traceback is {}", tb.c_str());
            if_(ocarina::has_nan(value) || ocarina::has_inf(value), [&] {
                $err_with_location("invalid value : ({}, {}, {}, {}) \n" + content, value);
                ret = make_float4(0.f);
            });
            return ret;
        }
    }
    static void set_printer_enabled(bool enabled) noexcept { printer().set_enabled(enabled); }
    [[nodiscard]] static bool valid_check() noexcept { return instance().valid_check_; }
    static void set_valid_check(bool val) noexcept { instance().valid_check_ = val; }
    [[nodiscard]] static bool code_obfuscation() noexcept { return instance().code_obfuscation_; }
    static void set_code_obfuscation(bool val) noexcept { instance().code_obfuscation_ = val; }
};

}// namespace ocarina