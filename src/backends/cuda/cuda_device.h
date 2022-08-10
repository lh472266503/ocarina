//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "rhi/device.h"
#include <cuda.h>
#include "util.h"

namespace ocarina {
class CUDADevice : public Device::Impl {
private:
    CUdevice _cu_device{};
    CUcontext _cu_ctx{};

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

    [[nodiscard]] ocarina::string get_ptx(const Function &func) const noexcept;

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
    [[nodiscard]] handle_ty create_buffer(size_t size) noexcept override;
    void destroy_buffer(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_texture(uint2 res, PixelStorage pixel_storage) noexcept override;
    void destroy_texture(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_shader(const Function &function) noexcept override;
    void destroy_shader(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_accel() noexcept override;
    void destroy_accel(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_stream() noexcept override;
    void destroy_stream(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_mesh(handle_ty v_handle, handle_ty t_handle,
                                        uint vert_num,
                                        uint v_stride, uint tri_num,
                                        AccelUsageTag usage_tag) noexcept override;
    virtual void destroy_mesh(handle_ty handle) noexcept override;
};
}// namespace ocarina
