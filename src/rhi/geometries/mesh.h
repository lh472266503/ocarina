//
// Created by Zero on 2022/8/8.
//

#pragma once

#include "rhi/resources/resource.h"
#include "rhi/command.h"
#include "dsl/struct.h"


namespace ocarina {


class Mesh : public RHIResource {
public:
    class Impl {
    protected:
        uint tri_num{};
        uint vertex_stride{};
        handle_ty vertices_handle{};
        handle_ty tri_handle{};
        friend class Mesh;
    };

public:
    Mesh(Device::Impl *device, handle_ty v_handle,
         handle_ty t_handle, uint t_num, uint v_stride, AccelUsageTag usage_tag = AccelUsageTag::FAST_TRACE)
        : RHIResource(device, Tag::MESH,
                      device->create_mesh(v_handle, t_handle, v_stride, t_num)) {}
//    template<typename T>
//    Mesh(Device::Impl *device, const Buffer<T> &v_buffer,
//         const Buffer<Triangle> &t_buffer, AccelUsageTag usage_tag = AccelUsageTag::FAST_TRACE)
//        : Mesh(device, v_buffer.handle(), t_buffer.handle(), t_buffer.size(), sizeof(T)) {}
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] uint triangle_num() const noexcept { return impl()->tri_num; }
    [[nodiscard]] handle_ty vertices_handle() const noexcept { return impl()->vertices_handle; }
    [[nodiscard]] handle_ty triangle_handle() const noexcept { return impl()->tri_handle; }
};

}// namespace ocarina

