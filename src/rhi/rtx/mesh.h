//
// Created by Zero on 2022/8/8.
//

#pragma once

#include "rhi/resources/resource.h"
#include "rhi/command.h"
#include "dsl/struct.h"
#include "../resources/buffer.h"

namespace ocarina {

class Mesh : public RHIResource {
public:
    class Impl {
        virtual handle_ty blas_handle() const noexcept = 0;
    };

public:
    Mesh(Device::Impl *device, const MeshParams &params)
        : RHIResource(device, Tag::MESH,
                      device->create_mesh(params)) {}

    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] MeshBuildCommand *build_bvh() noexcept {
        return MeshBuildCommand::create(_handle);
    }
};

template<typename Vertex, typename Tri>
Mesh Device::create_mesh(const Buffer<Vertex> &v_buffer, const Buffer<Tri> &t_buffer,
                         AccelUsageTag usage_tag, AccelGeomTag geom_tag) noexcept {
    MeshParams params;
    params.vert_handle = v_buffer.handle();
    params.vert_handle_ptr = v_buffer.handle_ptr();
    params.vert_stride = sizeof(Vertex);
    params.vert_num = v_buffer.size();

    params.tri_handle = t_buffer.handle();
    params.tri_num = t_buffer.size();
    params.tri_stride = sizeof(Tri);

    params.usage_tag = usage_tag;
    params.geom_tag = geom_tag;

    return _create<Mesh>(params);
}

}// namespace ocarina
