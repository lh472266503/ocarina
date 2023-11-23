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

public:
    [[nodiscard]] static Env &instance() noexcept;
    static void destroy_instance() noexcept;
    [[nodiscard]] Printer &printer() noexcept { return _printer; }
    [[nodiscard]] Debugger &debugger() noexcept { return _debugger; }
};

}// namespace ocarina