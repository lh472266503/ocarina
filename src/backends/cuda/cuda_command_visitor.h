//
// Created by zero on 2022/7/9.
//

#pragma once

#include "runtime/command.h"
#include <cuda.h>
namespace ocarina {

class CUDADevice;
class CUDACommandVisitor final : public CommandVisitor {
private:
    CUstream _stream{};
    CUDADevice *_device{};

public:
    explicit CUDACommandVisitor(CUstream stream, CUDADevice *device) : _stream(stream), _device(device) {}
    void visit(const BufferUploadCommand *cmd) noexcept override;
    void visit(const BufferDownloadCommand *cmd) noexcept override;
    void visit(const SynchronizeCommand *cmd) noexcept override;
};

}// namespace ocarina