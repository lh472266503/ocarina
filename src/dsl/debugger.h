//
// Created by Zero on 2023/11/17.
//

#pragma once

#include "core/basic_types.h"
#include "dsl/type_trait.h"
#include "var.h"
#include "builtin.h"
#include "syntax.h"

namespace ocarina {

class Debugger {
private:
    static Debugger *s_debugger;
    optional<Bool> _open{};
    optional<Uint2> _lower{};
    optional<Uint2> _upper{};

private:
    Debugger() = default;
    Debugger(const Debugger &) = default;
    Debugger(Debugger &&) = default;
    Debugger operator=(const Debugger &) = delete;
    Debugger operator=(Debugger &&) = delete;

public:
    [[nodiscard]] static Debugger &instance() noexcept;
    static void destroy_instance() noexcept;
    void switching(Bool open) noexcept;
    void reset() noexcept;

    template<typename Func>
    void execute(Func &&func) const noexcept {
        Uint2 pixel = dispatch_idx().xy();
        if constexpr (std::invocable<Func, Uint2>) {
            if_(all(pixel <= *_upper) && all(pixel >= *_lower), [&] {
                func(pixel);
            });
        } else {
            if_(all(pixel <= *_upper) && all(pixel >= *_lower), [&] {
                func();
            });
        }
    }
};

}// namespace ocarina