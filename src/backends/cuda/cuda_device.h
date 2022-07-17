//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "runtime/device.h"
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

public:
    explicit CUDADevice(Context *context);
    [[nodiscard]] handle_ty create_buffer(size_t size) noexcept override;
    template<typename Func>
    decltype(auto) bind_handle(Func &&func) noexcept {
        ContextGuard cg(_cu_ctx);
        return func();
    }
    template<typename Func>
    decltype(auto) bind_handle_sync(Func &&func) noexcept {
        std::mutex mutex;
        std::unique_lock lock(mutex);
        ContextGuard cg(_cu_ctx);
        return func();
    }
    void destroy_buffer(handle_ty handle) noexcept override;
    void destroy_texture(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_shader(ocarina::string_view str) noexcept override;
    [[nodiscard]] handle_ty create_stream() noexcept override;
    void destroy_stream(handle_ty handle) noexcept override;
};
}// namespace ocarina

