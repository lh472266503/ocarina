//
// Created by Zero on 2023/11/17.
//

#pragma once

#include "core/basic_types.h"
#include "dsl/type_trait.h"
#include "var.h"
#include "builtin.h"
#include "math/box.h"
#include "syntax.h"

namespace ocarina {
struct DebugData {
    Box2u range{};
    bool switching{};
};
}// namespace ocarina
OC_STRUCT(ocarina::DebugData, range, switching){};

namespace ocarina {

using OCDebugData = Var<DebugData>;

class Debugger {
private:
    static Debugger *s_debugger;
    optional<OCDebugData> _debug_data;

private:
    Debugger() = default;
    Debugger(const Debugger &) = default;
    Debugger(Debugger &&) = default;
    Debugger operator=(const Debugger &) = delete;
    Debugger operator=(Debugger &&) = delete;

public:
    [[nodiscard]] static Debugger &instance() noexcept;
    static void destroy_instance() noexcept;
    void reset() noexcept;
    void init(const OCDebugData &data) noexcept { _debug_data = data; }
    template<typename Func>
    void execute(Func &&func) const noexcept {
        if_(_debug_data->switching, [&] {
            Uint2 idx = dispatch_idx().xy();
            if_(_debug_data->range->contains(idx), [&] {
                if constexpr (std::invocable<Func, Uint2>) {
                    func(idx);
                } else {
                    func();
                }
            });
        });
    }
};

}// namespace ocarina