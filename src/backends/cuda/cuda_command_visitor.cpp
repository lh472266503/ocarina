//
// Created by zero on 2022/7/9.
//

#include "cuda_command_visitor.h"
#include "util.h"
#include "cuda_device.h"
#include "cuda_shader.h"

namespace ocarina {
void CUDACommandVisitor::visit(const BufferUploadCommand *cmd) noexcept {
    _device->use_context([&] {
        if (cmd->async()) {
            OC_CU_CHECK(cuMemcpyHtoDAsync(cmd->device_ptr(),
                                          cmd->host_ptr(),
                                          cmd->size_in_bytes(),
                                          _stream));
        } else {
            OC_CU_CHECK(cuMemcpyHtoD(cmd->device_ptr(),
                                     cmd->host_ptr(),
                                     cmd->size_in_bytes()));
        }
    });
}

void CUDACommandVisitor::visit(const BufferDownloadCommand *cmd) noexcept {
    _device->use_context([&] {
        if (cmd->async()) {
            OC_CU_CHECK(cuMemcpyDtoHAsync(cmd->host_ptr(),
                                          cmd->device_ptr(),
                                          cmd->size_in_bytes(),
                                          _stream));
        } else {
            OC_CU_CHECK(cuMemcpyDtoH(cmd->host_ptr(),
                                     cmd->device_ptr(),
                                     cmd->size_in_bytes()));
        }
    });
}

void CUDACommandVisitor::visit(const SynchronizeCommand *cmd) noexcept {
    OC_CU_CHECK(cuStreamSynchronize(_stream));
}

void CUDACommandVisitor::visit(const ShaderDispatchCommand *cmd) noexcept {
    uint3 grid_size = make_uint3(1);
    uint3 block_size = make_uint3(1);
    (reinterpret_cast<CUDAShader *>(cmd->entry()))->launch(handle_ty(_stream), const_cast<ShaderDispatchCommand*>(cmd));
}
void CUDACommandVisitor::visit(const TextureUploadCommand *cmd) noexcept {
}
void CUDACommandVisitor::visit(const TextureDownloadCommand *cmd) noexcept {

}

}// namespace ocarina