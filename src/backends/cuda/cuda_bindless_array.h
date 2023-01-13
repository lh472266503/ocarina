//
// Created by Zero on 12/01/2023.
//

#pragma once

#include "core/stl.h"
#include "rhi/managed.h"
#include "rhi/resources/bindless_array.h"
#include <cuda.h>

namespace ocarina {
class CUDADevice;
class CUDABindlessArray : public BindlessArray::Impl {
public:
    struct SlotSOA {
        CUdeviceptr buffer_slot;
        CUdeviceptr tex_slot;
    };

private:
    SlotSOA _slot_soa{};
    CUDADevice *_device{};
    Managed<CUdeviceptr> _buffers;
    Managed<CUtexObject> _textures;

public:
    explicit CUDABindlessArray(CUDADevice *device);
    [[nodiscard]] size_t size() const noexcept override { return sizeof(SlotSOA); }
    [[nodiscard]] size_t alignment() const noexcept override { return alignof(SlotSOA); }
    [[nodiscard]] size_t emplace_buffer(handle_ty handle) noexcept override;
    void remove_buffer(handle_ty index) noexcept override;
    [[nodiscard]] size_t emplace_texture(handle_ty handle) noexcept override;
    void remove_texture(handle_ty index) noexcept override;
    [[nodiscard]] BufferUploadCommand *upload_buffer_handles() const noexcept override;
    [[nodiscard]] BufferUploadCommand *upload_texture_handles() const noexcept override;
};

}// namespace ocarina