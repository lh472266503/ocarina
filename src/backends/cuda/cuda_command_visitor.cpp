//
// Created by zero on 2022/7/9.
//

#include "cuda_command_visitor.h"
#include "util.h"
#include "cuda_device.h"
#include "cuda_shader.h"
#include "cuda_mesh.h"

namespace ocarina {
void CUDACommandVisitor::visit(const BufferUploadCommand *cmd) noexcept {
    _device->use_context([&] {
        if (cmd->async()) {
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

void CUDACommandVisitor::visit(const BufferDownloadCommand *cmd) noexcept {
    _device->use_context([&] {
        if (cmd->async()) {
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

[[nodiscard]] CUDA_MEMCPY2D memcpy_desc(const TextureOpCommand *cmd) noexcept {
    CUDA_MEMCPY2D memcpy_desc{};
    memcpy_desc.srcXInBytes = 0;
    memcpy_desc.srcY = 0;
    memcpy_desc.srcPitch = cmd->width_in_bytes();
    memcpy_desc.dstPitch = cmd->width_in_bytes();
    memcpy_desc.dstArray = cmd->device_ptr<CUarray>();
    memcpy_desc.dstXInBytes = 0;
    memcpy_desc.dstY = 0;
    memcpy_desc.WidthInBytes = cmd->width_in_bytes();
    memcpy_desc.Height = cmd->height();
    return memcpy_desc;
}

}// namespace detail

void CUDACommandVisitor::visit(const TextureUploadCommand *cmd) noexcept {
    _device->use_context([&] {
        CUDA_MEMCPY2D desc = detail::memcpy_desc(cmd);
        desc.srcMemoryType = CU_MEMORYTYPE_HOST;
        desc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        desc.srcHost = cmd->host_ptr<const void *>();
        if (cmd->async()) {
            OC_CU_CHECK(cuMemcpy2DAsync(&desc, _stream));
        } else {
            OC_CU_CHECK(cuMemcpy2D(&desc));
        }
    });
}

void CUDACommandVisitor::visit(const TextureDownloadCommand *cmd) noexcept {
    _device->use_context([&] {
        CUDA_MEMCPY2D desc = detail::memcpy_desc(cmd);
        desc.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        desc.dstMemoryType = CU_MEMORYTYPE_HOST;
        desc.dstHost = reinterpret_cast<void *>(cmd->host_ptr());
        if (cmd->async()) {
            OC_CU_CHECK(cuMemcpy2DAsync(&desc, _stream));
        } else {
            OC_CU_CHECK(cuMemcpy2D(&desc));
        }
    });
}
void CUDACommandVisitor::visit(const MeshBuildCommand *cmd) noexcept {
    cmd->mesh<CUDAMesh>()->build_bvh(cmd);
}
void CUDACommandVisitor::visit(const AccelBuildCommand *cmd) noexcept {
}

}// namespace ocarina