//
// Created by zero on 2022/7/9.
//

#pragma once

#include "runtime/command.h"
#include <cuda.h>
namespace ocarina {

class CUDACommandVisitor final : public CommandVisitor {
private:
    CUstream _stream{};

public:
    explicit CUDACommandVisitor(CUstream stream) : _stream(stream) {}
    void visit(const BufferUploadCommand *cmd) noexcept override;
    void visit(const BufferDownloadCommand *cmd) noexcept override;
    void visit(const SynchronizeCommand *cmd) noexcept override;
};

}// namespace ocarina