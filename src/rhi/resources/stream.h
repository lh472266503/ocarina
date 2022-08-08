//
// Created by Zero on 06/07/2022.
//

#pragma once

#include "resource.h"
#include "rhi/device.h"
#include "rhi/command_queue.h"

namespace ocarina {

struct Commit {
    using func_ty = std::function<void(void *)>;
    func_ty callback;
    Commit() = default;
    explicit Commit(func_ty &&func) : callback(std::forward<func_ty>(func)) {}
};

template<typename... Args>
[[nodiscard]] Commit commit(Args &&...args) {
    return Commit(OC_FORWARD(args)...);
}

class Stream : public RHIResource {
public:
    class Impl {
    protected:
        CommandQueue _command_queue;

    public:
        virtual void barrier() noexcept = 0;
        virtual void commit(const Commit &commit) noexcept = 0;
        virtual void add_command(Command *cmd) noexcept = 0;
    };

public:
    explicit Stream(Device::Impl *device);
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    Stream &operator<<(Command *command) noexcept;
    Stream &operator<<(const Commit &commit) noexcept;
    void commit(const Commit &cmt) noexcept;
};

}// namespace ocarina