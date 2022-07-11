//
// Created by Zero on 06/07/2022.
//

#pragma once

#include "resource.h"
#include "device.h"
#include "command_queue.h"

namespace ocarina {

class Stream : public Resource {
public:
    class Impl {
    protected:
        CommandQueue _command_queue;

    public:
        virtual void synchronize() noexcept = 0;
        virtual void barrier() noexcept = 0;
        virtual void commit() noexcept = 0;
        virtual void add_command(Command *cmd) noexcept = 0;
    };

public:
    explicit Stream(Device::Impl *device);
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    Stream &operator<<(Command *command) noexcept;
    Stream &operator<<(CommandQueue::Synchronize) noexcept;
    Stream &operator<<(CommandQueue::Commit) noexcept;
    Stream &synchronize() noexcept;
    void commit() noexcept;
};

}// namespace ocarina