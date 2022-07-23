//
// Created by Zero on 03/07/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include "device.h"
namespace ocarina {

using handle_ty = uint64_t;
using ptr_t = uint64_t;
class Resource  {
public:
    enum Tag : uint8_t {
        BUFFER,
        TEXTURE,
        STREAM,
        SHADER
    };

protected:
    Tag _tag;
    handle_ty _handle{};
    Device::Impl *_device{nullptr};

protected:
    void _destroy();

public:
    Resource(Device::Impl *device, Tag tag, handle_ty handle)
        : _device(device), _tag(tag), _handle(handle) {}
    [[nodiscard]] Tag tag() const noexcept { return _tag; }
    [[nodiscard]] handle_ty handle() const noexcept { return _handle; }
    [[nodiscard]] handle_ty *handle_address() noexcept { return &_handle; }
    ~Resource() {
        _destroy();
    }
};
}// namespace ocarina