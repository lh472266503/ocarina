//
// Created by Zero on 2023/11/17.
//

#pragma once

#include "core/basic_types.h"
#include "dsl/type_trait.h"
#include "var.h"
#include "builtin.h"
#include "rhi/resources/managed.h"
#include "math/box.h"
#include "syntax.h"

namespace ocarina {
struct DebugData {
    Box2u range{};
    int enabled{1};
};
}// namespace ocarina

// clang-format off
OC_STRUCT(ocarina::DebugData, range, enabled){};
// clang-format on

namespace ocarina {

using OCDebugData = Var<DebugData>;

class Debugger {
private:
    static Debugger *s_debugger;
    Managed<DebugData> _data;

private:
    Debugger() = default;
    Debugger(const Debugger &) = delete;
    Debugger(Debugger &&) = delete;
    Debugger operator=(const Debugger &) = delete;
    Debugger operator=(Debugger &&) = delete;

public:
    [[nodiscard]] static Debugger &instance() noexcept;
    static void destroy_instance() noexcept;
    void reset() noexcept { _data[0] = DebugData{}; }
    void init(Device &device) noexcept { _data.reset_all(device, 1); }
    [[nodiscard]] auto &host_data() const noexcept { return _data.host_buffer()[0]; }
    [[nodiscard]] auto &host_data() noexcept { return _data.host_buffer()[0]; }
    [[nodiscard]] Command *upload(bool async = true) const noexcept { return _data.upload(async); }
    void upload_immediately() const noexcept { _data.upload_immediately(); }
    void filp_enabled() noexcept { host_data().enabled = !host_data().enabled; }
    [[nodiscard]] bool is_enabled() const noexcept { return host_data().enabled; }
    void disable() noexcept { host_data().enabled = false; }
    void reset_range() noexcept { host_data().range = Box2u{}; }
    void set_lower(uint2 lower) noexcept { host_data().range.extend(lower); }
    void set_upper(uint2 upper) noexcept { host_data().range.extend(upper); }

protected:
    /// for dsl
    [[nodiscard]] Bool _is_enabled() const noexcept { return cast<bool>(_data.read(0).enabled); }
    /// for dsl
    [[nodiscard]] OCDebugData _device_data() const noexcept { return _data.read(0); }

public:
    /// for dsl
    template<typename Func>
    void execute(Func &&func) const noexcept {
        if_(_is_enabled(), [&] {
            Uint2 idx = dispatch_idx().xy();
            if_(_device_data()->range->contains(idx), [&] {
                if constexpr (std::invocable<Func, Uint2>) {
                    func(idx);
                } else {
                    func();
                }
            });
        });
    }

    template<typename Func>
    void operator*(Func &&func) const noexcept {
        execute(OC_FORWARD(func));
    }
};

}// namespace ocarina