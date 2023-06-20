//
// Created by zero on 2022/7/9.
//

#include "cuda_command_visitor.h"
#include "util.h"
#include "cuda_device.h"
#include "cuda_shader.h"
#include "cuda_mesh.h"
#include "optix_accel.h"

namespace ocarina {
void CUDACommandVisitor::visit(const BufferUploadCommand *cmd) noexcept {
    _device->use_context([&] {
        if (cmd->async() && _stream) {
            OC_CU_CHECK(cuMemcpyHtoDAsync(cmd->device_ptr(),
                                          cmd->host_ptr<const void *>(),
                                          cmd->size_in_bytes(),
                                          _stream));
        } else {
            OC_CU_CHECK(cuMemcpyHtoD(cmd->device_ptr(),
                                     cmd->host_ptr<const void *>(),
                                     cmd->size_in_bytes()));
        }
    });
}

void CUDACommandVisitor::visit(const BufferByteSetCommand *cmd) noexcept {
    _device->use_context([&] {
        if (cmd->async() && _stream) {
            OC_CU_CHECK(cuMemsetD8Async(cmd->device_ptr(), cmd->value(),
                                        cmd->size_in_bytes(), _stream));
        } else {
            OC_CU_CHECK(cuMemsetD8(cmd->device_ptr(), cmd->value(),
                                   cmd->size_in_bytes()));
        }
    });
}

void CUDACommandVisitor::visit(const BufferCopyCommand *cmd) noexcept {
    _device->use_context([&] {
        auto src_buffer = cmd->src() + cmd->src_offset();
        auto dst_buffer = cmd->dst() + cmd->dst_offset();
        if (cmd->async() && _stream) {
            OC_CU_CHECK(cuMemcpyDtoD(dst_buffer, src_buffer, cmd->size()));
        } else {
            OC_CU_CHECK(cuMemcpyDtoDAsync(dst_buffer, src_buffer, cmd->size(), _stream));
        }
    });
}

void CUDACommandVisitor::visit(const BufferDownloadCommand *cmd) noexcept {
    _device->use_context([&] {
        if (cmd->async() && _stream) {
            OC_CU_CHECK(cuMemcpyDtoHAsync(cmd->host_ptr<void *>(),
                                          cmd->device_ptr(),
                                          cmd->size_in_bytes(),
                                          _stream));
        } else {
            OC_CU_CHECK(cuMemcpyDtoH(cmd->host_ptr<void *>(),
                                     cmd->device_ptr(),
                                     cmd->size_in_bytes()));
        }
    });
}

void CUDACommandVisitor::visit(const SynchronizeCommand *cmd) noexcept {
    OC_CU_CHECK(cuStreamSynchronize(_stream));
}

void CUDACommandVisitor::visit(const ShaderDispatchCommand *cmd) noexcept {
    cmd->entry<CUDAShader *>()->launch(handle_ty(_stream),
                                       const_cast<ShaderDispatchCommand *>(cmd));
}

namespace detail {

[[nodiscard]] CUDA_MEMCPY3D memcpy_desc(const TextureOpCommand *cmd) noexcept {
    CUDA_MEMCPY3D memcpy_desc{};
    memcpy_desc.srcXInBytes = 0;
    memcpy_desc.srcY = 0;
    memcpy_desc.srcPitch = cmd->width_in_bytes();
    memcpy_desc.dstPitch = cmd->width_in_bytes();
    memcpy_desc.dstXInBytes = 0;
    memcpy_desc.dstY = 0;
    memcpy_desc.WidthInBytes = cmd->width_in_bytes();
    memcpy_desc.Height = cmd->height();
    memcpy_desc.Depth = cmd->depth();
    return memcpy_desc;
}

}// namespace detail

void CUDACommandVisitor::visit(const TextureUploadCommand *cmd) noexcept {
    _device->use_context([&] {
        CUDA_MEMCPY3D desc = detail::memcpy_desc(cmd);
        desc.srcMemoryType = CU_MEMORYTYPE_HOST;
        desc.srcHost = cmd->host_ptr<const void *>();
        desc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        desc.dstArray = cmd->device_ptr<CUarray>();
        if (cmd->async() && _stream) {
            OC_CU_CHECK(cuMemcpy3DAsync(&desc, _stream));
        } else {
            OC_CU_CHECK(cuMemcpy3D(&desc));
        }
    });
}

void CUDACommandVisitor::visit(const TextureDownloadCommand *cmd) noexcept {
    _device->use_context([&] {
        CUDA_MEMCPY3D desc = detail::memcpy_desc(cmd);
        desc.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        desc.dstMemoryType = CU_MEMORYTYPE_HOST;
        desc.srcArray = cmd->device_ptr<CUarray>();
        desc.dstHost = reinterpret_cast<void *>(cmd->host_ptr());
        if (cmd->async() && _stream) {
            OC_CU_CHECK(cuMemcpy3DAsync(&desc, _stream));
        } else {
            OC_CU_CHECK(cuMemcpy3D(&desc));
        }
    });
}

void CUDACommandVisitor::visit(const TextureCopyCommand *cmd) noexcept {
    _device->use_context([&] {
        CUDA_MEMCPY3D copy{};
        uint pitch = pixel_size(cmd->pixel_storage()) * cmd->resolution().x;
        copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.srcArray = cmd->src<CUarray>();
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = cmd->dst<CUarray>();
        copy.WidthInBytes = pitch;
        copy.Height = cmd->resolution().y;
        copy.Depth = cmd->resolution().z;
        if (cmd->async() && _stream) {
            OC_CU_CHECK(cuMemcpy3DAsync(&copy, _stream));
        } else {
            OC_CU_CHECK(cuMemcpy3D(&copy));
        }
    });
}

void CUDACommandVisitor::visit(const ocarina::BufferToTextureCommand *cmd) noexcept {
    _device->use_context([&] {
        CUDA_MEMCPY3D copy{};
        uint pitch = pixel_size(cmd->pixel_storage()) * cmd->resolution().x;
        copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.srcDevice = cmd->src() + cmd->buffer_offset();
        copy.srcPitch = pitch;
        copy.srcHeight = cmd->resolution().y;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = cmd->dst<CUarray>();
        copy.WidthInBytes = pitch;
        copy.Height = cmd->resolution().y;
        copy.Depth = cmd->resolution().z;
        if (cmd->async() && _stream) {
            OC_CU_CHECK(cuMemcpy3DAsync(&copy, _stream));
        } else {
            OC_CU_CHECK(cuMemcpy3D(&copy));
        }
    });
}

void CUDACommandVisitor::visit(const MeshBuildCommand *cmd) noexcept {
    cmd->mesh<CUDAMesh>()->build_bvh(cmd);
}
void CUDACommandVisitor::visit(const AccelBuildCommand *cmd) noexcept {
    cmd->accel<OptixAccel>()->build_bvh(this);
}

}// namespace ocarina