//
// Created by zero on 2022/7/9.
//

#include "cuda_command_visitor.h"
#include "util.h"

namespace ocarina {
void CUDACommandVisitor::visit(const BufferUploadCommand *cmd) noexcept {
    OC_CU_CHECK(cuMemcpyHtoDAsync(cmd->device_ptr(),
                                  cmd->host_ptr(),
                                  cmd->size_in_bytes(),
                                  _stream));
}

void CUDACommandVisitor::visit(const BufferDownloadCommand *cmd) noexcept {
    OC_CU_CHECK(cuMemcpyDtoHAsync(cmd->host_ptr(),
                                  cmd->device_ptr(),
                                  cmd->size_in_bytes(),
                                  _stream));
}

void CUDACommandVisitor::visit(const SynchronizeCommand *cmd) noexcept {
    OC_CU_CHECK(cuStreamSynchronize(_stream));
}

}// namespace ocarina