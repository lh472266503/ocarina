//
// Created by Zero on 2022/8/8.
//

#pragma once

#include "rhi/resources/resource.h"
#include "rhi/command.h"
#include "mesh.h"

namespace ocarina {
class Accel : public RHIResource {
public:
    class Impl {
    public:
        virtual void add_mesh(Mesh::Impl *mesh, float4x4 transform) noexcept = 0;
    };
public:
    explicit Accel(Device::Impl *device)
        : RHIResource(device, Tag::ACCEL, device->create_accel()) {}

    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    void add_mesh(Mesh &&mesh, float4x4 transform) noexcept {
        impl()->add_mesh(mesh.impl(), transform);
    }

};
}// namespace ocarina
