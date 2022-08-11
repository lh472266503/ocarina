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
private:
    ocarina::vector<Mesh> _meshes;
    ocarina::vector<float4x4> _transforms;
public:
    explicit Accel(Device::Impl *device)
        : RHIResource(device, Tag::ACCEL, device->create_accel()) {}

    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    void add_mesh(Mesh &&mesh, float4x4 transform) noexcept {
        _meshes.push_back(std::move(mesh));
        _transforms.push_back(transform);
    }

};
}// namespace ocarina
