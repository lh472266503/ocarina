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
private:
    uint _triangle_num{};
    uint _vertex_num{};
    vector<RHIMesh> _meshes;

public:
    class Impl {
    public:
        virtual void add_mesh(const RHIMesh::Impl *mesh, float4x4 transform) noexcept = 0;
        [[nodiscard]] virtual handle_ty handle() const noexcept = 0;
        [[nodiscard]] virtual const void *handle_ptr() const noexcept = 0;
        [[nodiscard]] virtual size_t data_size() const noexcept = 0;
        [[nodiscard]] virtual void clear() noexcept = 0;
        [[nodiscard]] virtual size_t data_alignment() const noexcept = 0;
    };

public:
    Accel() = default;
    explicit Accel(Device::Impl *device)
        : RHIResource(device, Tag::ACCEL, device->create_accel()) {}
    [[nodiscard]] uint triangle_num() const noexcept { return _triangle_num; }
    [[nodiscard]] uint vertex_num() const noexcept { return _vertex_num; }
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }

    void add_mesh(RHIMesh m, float4x4 transform) noexcept {
        _meshes.push_back(ocarina::move(m));
        RHIMesh &mesh = _meshes.back();
        _vertex_num += mesh.vertex_num();
        _triangle_num += mesh.triangle_num();
        impl()->add_mesh(mesh.impl(), transform);
    }

    void clear() noexcept {
        impl()->clear();
        _meshes.clear();
        _triangle_num = 0;
        _vertex_num = 0;
    }

    [[nodiscard]] const Expression *expression() const noexcept override {
        const ArgumentBinding &uniform = Function::current()->get_uniform_var(Type::of<decltype(*this)>(),
                                                                              Variable::Tag::ACCEL,
                                                                              memory_block());
        return uniform.expression();
    }

    template<typename TRay>
    [[nodiscard]] Var<bool> trace_any(const TRay &ray) const noexcept {
        return make_expr<Accel>(expression()).trace_any(ray);
    }

    template<typename TRay>
    [[nodiscard]] Var<Hit> trace_closest(const TRay &ray) const noexcept {
        return make_expr<Accel>(expression()).trace_closest(ray);
    }

    [[nodiscard]] handle_ty handle() const noexcept override { return impl()->handle(); }
    [[nodiscard]] const void *handle_ptr() const noexcept override { return impl()->handle_ptr(); }
    [[nodiscard]] TLASBuildCommand *build_bvh() noexcept {
        return TLASBuildCommand::create(_handle);
    }
};
}// namespace ocarina
