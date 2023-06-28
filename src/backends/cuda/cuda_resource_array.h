//
// Created by Zero on 12/01/2023.
//

#pragma once

#include "core/stl.h"
#include "rhi/resources/managed.h"
#include "rhi/resources/resource_array.h"
#include <cuda.h>

namespace ocarina {
class CUDADevice;
class CUDAResourceArray : public ResourceArray::Impl {

private:
    SlotSOA _slot_soa{};
    CUDADevice *_device{};
    Managed<BufferDesc> _buffers;
    Managed<CUtexObject> _textures;

public:
    explicit CUDAResourceArray(CUDADevice *device);

    /// for device side structure
    [[nodiscard]] const void *handle_ptr() const noexcept override {
        return &_slot_soa;
    }
    [[nodiscard]] size_t max_member_size() const noexcept override { return sizeof(CUdeviceptr); }
    [[nodiscard]] size_t data_size() const noexcept override { return sizeof(SlotSOA); }
    [[nodiscard]] size_t data_alignment() const noexcept override { return alignof(SlotSOA); }
    void prepare_slotSOA(Device &device) noexcept override;

    [[nodiscard]] size_t emplace_buffer(handle_ty handle,size_t size_in_byte) noexcept override;
    void remove_buffer(handle_ty index) noexcept override;
    [[nodiscard]] size_t emplace_texture(handle_ty handle) noexcept override;
    void remove_texture(handle_ty index) noexcept override;
    void set_buffer(ocarina::handle_ty index, ocarina::handle_ty handle, size_t size_in_byte) noexcept override;
    void set_texture(ocarina::handle_ty index, ocarina::handle_ty handle) noexcept override;
    [[nodiscard]] size_t buffer_num() const noexcept override;
    [[nodiscard]] size_t texture_num() const noexcept override;
    [[nodiscard]] BufferUploadCommand *upload_buffer_handles() const noexcept override;
    [[nodiscard]] BufferUploadCommand *upload_texture_handles() const noexcept override;
    [[nodiscard]] BufferUploadCommand *upload_buffer_handles_sync() const noexcept override;
    [[nodiscard]] BufferUploadCommand *upload_texture_handles_sync() const noexcept override;
};

}// namespace ocarina