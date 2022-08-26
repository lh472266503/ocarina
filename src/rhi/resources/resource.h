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
    RHIResource() = default;

    RHIResource(RHIResource &&other) noexcept {
        if (&other == this) { return; }
        _tag = other._tag;
        _device = other._device;
        _handle = other._handle;
        other._device = nullptr;
    }

    RHIResource &operator=(RHIResource &&other) noexcept {
        if (&other == this) { return *this; }
        _tag = other._tag;
        _device = other._device;
        _handle = other._handle;
        other._device = nullptr;
        return *this;
    }

    [[nodiscard]] Tag tag() const noexcept { return _tag; }
    [[nodiscard]] virtual handle_ty handle() const noexcept { return _handle; }
    [[nodiscard]] virtual const void *handle_ptr() const noexcept { return &_handle; }
    // size of data on device side
    [[nodiscard]] virtual size_t data_size() const noexcept { return sizeof(handle_ty); }
    // alignment of data on device side
    [[nodiscard]] virtual size_t data_alignment() const noexcept { return sizeof(handle_ty); }
    [[nodiscard]] virtual size_t max_member_size() const noexcept { return sizeof(handle_ty); }
    [[nodiscard]] virtual MemoryBlock memory_block() const noexcept {
        return {handle_ptr(), data_size(), data_alignment(), max_member_size()};
    }
    [[nodiscard]] bool valid() const noexcept { return bool(_device); }
    void destroy();
    virtual ~RHIResource() { destroy(); }
};
}// namespace ocarina