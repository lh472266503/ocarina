//
// Created by Zero on 2022/8/8.
//

#pragma once

#include "rhi/resources/resource.h"
#include "rhi/command.h"
#include "mesh.h"
#include "dsl/var.h"

namespace ocarina {
class OC_RHI_API Accel : public RHIResource {
private:
    uint triangle_num_{};
    uint vertex_num_{};

public:
    class Impl {
    protected:
        vector<RHIMesh> meshes_;
        vector<float4x4> transforms_;

    public:
        virtual void add_instance(RHIMesh mesh, float4x4 transform) noexcept {
            meshes_.push_back(ocarina::move(mesh));
            transforms_.push_back(transform);
        }
        [[nodiscard]] virtual handle_ty handle() const noexcept = 0;
        [[nodiscard]] virtual const void *handle_ptr() const noexcept = 0;
        [[nodiscard]] size_t mesh_num() const noexcept { return meshes_.size(); }
        [[nodiscard]] virtual size_t data_size() const noexcept = 0;
        virtual void clear() noexcept {
            meshes_.clear();
            transforms_.clear();
        }
        [[nodiscard]] virtual size_t data_alignment() const noexcept = 0;
    };

public:
    Accel() = default;
    explicit Accel(Device::Impl *device)
        : RHIResource(device, Tag::ACCEL, device->create_accel()) {}
    [[nodiscard]] uint triangle_num() const noexcept { return triangle_num_; }
    [[nodiscard]] uint vertex_num() const noexcept { return vertex_num_; }
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(handle_); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(handle_); }

    void add_instance(RHIMesh mesh, float4x4 transform) noexcept {
        triangle_num_ += mesh.triangle_num();
        vertex_num_ += mesh.vertex_num();
        impl()->add_instance(ocarina::move(mesh), transform);
    }

    [[nodiscard]] size_t mesh_num() const noexcept { return impl()->mesh_num(); }

    void clear() noexcept {
        if (impl()) {
            impl()->clear();
        }
        triangle_num_ = 0;
        vertex_num_ = 0;
    }

    [[nodiscard]] const Expression *expression() const noexcept override {
        const CapturedResource &captured_resource = Function::current()->get_captured_resource(Type::of<decltype(*this)>(),
                                                                               Variable::Tag::ACCEL,
                                                                               memory_block());
        return captured_resource.expression();
    }

    template<typename TRay>
    [[nodiscard]] Var<bool> trace_occlusion(const TRay &ray) const noexcept {
        return make_expr<Accel>(expression()).trace_occlusion(ray);
    }

    template<typename TRay>
    [[nodiscard]] Var<TriangleHit> trace_closest(const TRay &ray) const noexcept {
        return make_expr<Accel>(expression()).trace_closest(ray);
    }

    [[nodiscard]] handle_ty handle() const noexcept override { return impl()->handle(); }
    [[nodiscard]] const void *handle_ptr() const noexcept override { return impl()->handle_ptr(); }
    [[nodiscard]] TLASBuildCommand *build_bvh() noexcept {
        return TLASBuildCommand::create(handle_);
    }
};
}// namespace ocarina
