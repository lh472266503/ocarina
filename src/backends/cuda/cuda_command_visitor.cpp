//
// Created by zero on 2022/7/9.
//

#include "cuda_commad_visitor.h"

namespace ocarina {
void CUDACommandVisitor::visit(const BufferUploadCommand *cmd) noexcept {
}

void CUDACommandVisitor::visit(const BufferDownloadCommand *cmd) noexcept {
}

}// namespace ocarina