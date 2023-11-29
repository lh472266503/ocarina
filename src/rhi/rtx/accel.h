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

public:
    class Impl {
    protected:
        vector<RHIMesh> _meshes;
        vector<float4x4> _transforms;

    public:
        virtual void add_instance(RHIMesh mesh, float4x4 transform) noexcept {
            _meshes.push_back(ocarina::move(mesh));
            _transforms.push_back(transform);
        }
        [[nodiscard]] virtual handle_ty handle() const noexcept = 0;
        [[nodiscard]] virtual const void *handle_ptr() const noexcept = 0;
        [[nodiscard]] virtual size_t data_size() const noexcept = 0;
        virtual void clear() noexcept {
            _meshes.clear();
            _transforms.clear();
        }
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

    void add_instance(RHIMesh mesh, float4x4 transform) noexcept {
        _triangle_num += mesh.triangle_num();
        _vertex_num += mesh.vertex_num();
        impl()->add_instance(ocarina::move(mesh), transform);
    }

    void clear() noexcept {
        if (impl()) {
            impl()->clear();
        }
        _triangle_num = 0;
        _vertex_num = 0;
    }

    [[nodiscard]] const Expression *expression() const noexcept override {
        const ArgumentBinding &uniform = Function::current()->get_captured_var(Type::of<decltype(*this)>(),
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
