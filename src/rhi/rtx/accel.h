//
// Created by Zero on 2022/8/8.
//

#pragma once

#include "rhi/resources/resource.h"
#include "rhi/command.h"
#include "mesh.h"
#include "dsl/var.h"

namespace ocarina {
class Accel : public RHIResource {
public:
    class Impl {
    public:
        virtual void add_mesh(const Mesh::Impl *mesh, float4x4 transform) noexcept = 0;
        [[nodiscard]] virtual handle_ty handle() const noexcept = 0;
        [[nodiscard]] virtual const void *handle_ptr() const noexcept = 0;
    };

public:
    explicit Accel(Device::Impl *device)
        : RHIResource(device, Tag::ACCEL, device->create_accel()) {}

    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    void add_mesh(const Mesh &mesh, float4x4 transform) noexcept {
        impl()->add_mesh(mesh.impl(), transform);
    }

    template<typename TRay>
    [[nodiscard]] Var<bool> trace_any(const TRay &ray) const noexcept {
        const UniformBinding &uniform = Function::current()->get_uniform_var(Type::of<Accel>(),
                                                                             handle_ptr(),
                                                                             Variable::Tag::ACCEL);
        return make_expr<Accel>(uniform.expression()).trace_any(ray);
    }

    template<typename TRay>
    [[nodiscard]] Var<Hit> trace_closest(const TRay &ray) const noexcept {
        const UniformBinding &uniform = Function::current()->get_uniform_var(Type::of<Accel>(),
                                                                             handle_ptr(),
                                                                             Variable::Tag::ACCEL,
                                                                             data_size());
        return make_expr<Accel>(uniform.expression()).trace_closest(ray);
    }

    [[nodiscard]] handle_ty handle() const noexcept override {
        return impl()->handle();
    }
    [[nodiscard]] const void *handle_ptr() const noexcept override {
        return impl()->handle_ptr();
    }
    [[nodiscard]] AccelBuildCommand *build_bvh() noexcept {
        return AccelBuildCommand::create(_handle);
    }
};
}// namespace ocarina
