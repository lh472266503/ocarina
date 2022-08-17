//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "rhi/device.h"
#include <cuda.h>
#include "util.h"
#include "cuda_compiler.h"

namespace ocarina {
class CUDADevice : public Device::Impl {
private:
    CUdevice _cu_device{};
    CUcontext _cu_ctx{};
    OptixDeviceContext _optix_device_context{};
    OptixPipeline _optix_pipeline{};

    class ContextGuard {
    private:
        CUcontext _ctx{};

    public:
        explicit ContextGuard(CUcontext handle)
            : _ctx(handle) {
            OC_CU_CHECK(cuCtxPushCurrent(_ctx));
        }
        ~ContextGuard() {
            CUcontext ctx = nullptr;
            OC_CU_CHECK(cuCtxPopCurrent(&ctx));
            if (ctx != _ctx) [[unlikely]] {
                OC_ERROR_FORMAT(
                    "Invalid CUDA context {} (expected {}).",
                    fmt::ptr(ctx), fmt::ptr(_ctx));
            }
        }
    };

public:
    explicit CUDADevice(Context *context);
    template<typename Func>
    decltype(auto) use_context(Func &&func) noexcept {
        ContextGuard cg(_cu_ctx);
        return func();
    }
    template<typename Func>
    decltype(auto) use_context_sync(Func &&func) noexcept {
        std::mutex mutex;
        std::unique_lock lock(mutex);
        ContextGuard cg(_cu_ctx);
        return func();
    }

    template<typename T>
    void download(T *host_ptr, CUdeviceptr device_ptr, size_t num = 1, size_t offset = 0) {
        use_context([&] {
            OC_CU_CHECK(cuMemcpyDtoH(host_ptr, device_ptr + offset * sizeof(T), num * sizeof(T)));
        });
    }
    template<typename T>
    [[nodiscard]] T download(CUdeviceptr device_ptr, size_t offset = 0) {
        T ret;
        download<T>(&ret, device_ptr, 1, offset);
        return ret;
    }
    void init_optix_context() noexcept;
    [[nodiscard]] OptixDeviceContext optix_device_context() const noexcept { return _optix_device_context; }
    [[nodiscard]] handle_ty create_buffer(size_t size) noexcept override;
    void destroy_buffer(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_image(uint2 res, PixelStorage pixel_storage) noexcept override;
    void destroy_texture(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_shader(const Function &function) noexcept override;
    void destroy_shader(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_accel() noexcept override;
    void destroy_accel(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_stream() noexcept override;
    void destroy_stream(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_mesh(const MeshParams &params) noexcept override;
    void destroy_mesh(handle_ty handle) noexcept override;
    void init_rtx() noexcept override { init_optix_context(); }
};
}// namespace ocarina
