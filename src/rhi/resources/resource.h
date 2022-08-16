//
// Created by Zero on 03/07/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include "rhi/device.h"

namespace ocarina {

using handle_ty = uint64_t;
using ptr_t = uint64_t;

class RHIResource : public concepts::Noncopyable {
public:
    enum Tag : uint8_t {
        BUFFER,
        TEXTURE,
        STREAM,
        SHADER,
        MESH,
        ACCEL,
    };

protected:
    Tag _tag{};
    handle_ty _handle{};
    Device::Impl *_device{nullptr};

protected:
    RHIResource(Device::Impl *device, Tag tag, handle_ty handle)
        : _device(device), _tag(tag), _handle(handle) {}

public:
    RHIResource(RHIResource &&other) noexcept {
        if (&other == this) { return; }
        _tag = other._tag;
        _device = other._device;
        _handle = other._handle;
        other._device = nullptr;
    }
    [[nodiscard]] Tag tag() const noexcept { return _tag; }
    [[nodiscard]] handle_ty handle() const noexcept { return _handle; }
    [[nodiscard]] virtual const void *handle_ptr() const noexcept { return &_handle; }
    [[nodiscard]] bool valid() const noexcept { return bool(_device); }
    void destroy();
    virtual ~RHIResource() { destroy(); }
};
}// namespace ocarina