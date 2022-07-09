//
// Created by Zero on 06/07/2022.
//

#pragma once

#include "resource.h"
#include "device.h"
namespace ocarina {

class Command;

class Stream : public Resource {
public:
    class Impl {
        virtual void synchronize() noexcept = 0;
        virtual void barrier() noexcept = 0;
        virtual void flush() noexcept = 0;
    };

private:
    ocarina::vector<Command *> _commands;
    Impl *_impl{};

public:
    explicit Stream(Device::Impl *device)
        : Resource(device, Tag::STREAM,
                   device->create_stream()) {}

    Stream &operator<<(Command *command) noexcept {

    }

    Stream &synchronize() noexcept {

    }

    void flush() noexcept {

    }
};

}// namespace ocarina