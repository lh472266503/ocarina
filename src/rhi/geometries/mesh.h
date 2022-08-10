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
    protected:
        uint tri_num{};
        uint vertex_stride{};
        AccelUsageTag usage_tag;
        friend class Mesh;

    public:
        Impl(uint tri_num, uint v_stride, AccelUsageTag usage_tag)
            : tri_num(tri_num), vertex_stride(v_stride), usage_tag(usage_tag) {}
    };

public:
    Mesh(Device::Impl *device, handle_ty v_handle,
         handle_ty t_handle, uint t_num, uint v_stride, AccelUsageTag usage_tag)
        : RHIResource(device, Tag::MESH,
                      device->create_mesh(v_handle, t_handle, v_stride, t_num, usage_tag)) {}

    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] uint triangle_num() const noexcept { return impl()->tri_num; }
};

template<typename Vertex, typename Tri>
Mesh Device::create_mesh(const Buffer<Vertex> &v_buffer, const Buffer<Tri> &t_buffer, AccelUsageTag usage_tag) noexcept {
    return _create<Mesh>(v_buffer.handle(), t_buffer.handle(), t_buffer.size(), sizeof(Vertex), usage_tag);
}

}// namespace ocarina
