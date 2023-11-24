//
// Created by Zero on 11/05/2022.
//

#pragma once

#include "builtin.h"
#include "type_trait.h"
#include "expr.h"
#include "var.h"
#include "array.h"
#include "stmt_builder.h"
#include "syntax.h"
#include "operators.h"
#include "func.h"
#include "struct.h"
#include "rtx_type.h"
#include "computable.inl"
#include "polymorphic.h"
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
    static void set_printer_enabled(bool enabled) noexcept { printer().set_enabled(enabled); }
    [[nodiscard]] static bool valid_check() noexcept { return instance()._valid_check; }
    static void set_valid_check(bool val) noexcept { instance()._valid_check = val; }
    [[nodiscard]] static bool code_obfuscation() noexcept { return instance()._code_obfuscation; }
    static void set_code_obfuscation(bool val) noexcept { instance()._code_obfuscation = val; }
};

}// namespace ocarina