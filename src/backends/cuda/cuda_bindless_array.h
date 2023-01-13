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

    /// for device side structure
    [[nodiscard]] const void *handle_ptr() const noexcept override { return &_slot_soa; }
    [[nodiscard]] size_t max_member_size() const noexcept override { return sizeof(CUdeviceptr); }
    [[nodiscard]] size_t data_size() const noexcept override { return sizeof(SlotSOA); }
    [[nodiscard]] size_t data_alignment() const noexcept override { return alignof(SlotSOA); }
    void prepare_slotSOA() noexcept override;

    [[nodiscard]] size_t emplace_buffer(handle_ty handle) noexcept override;
    void remove_buffer(handle_ty index) noexcept override;
    [[nodiscard]] size_t emplace_texture(handle_ty handle) noexcept override;
    void remove_texture(handle_ty index) noexcept override;
    [[nodiscard]] BufferUploadCommand *upload_buffer_handles() const noexcept override;
    [[nodiscard]] BufferUploadCommand *upload_texture_handles() const noexcept override;
};

}// namespace ocarina