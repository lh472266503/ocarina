//
// Created by zero on 2022/7/9.
//

#pragma once

#include "rhi/command.h"
#include <cuda.h>
namespace ocarina {

class CUDADevice;
class CUDACommandVisitor final : public CommandVisitor {
private:
    CUDADevice *_device{};
    CUstream _stream{};

public:
    explicit CUDACommandVisitor(CUDADevice *device, CUstream stream = nullptr)
        : _device(device),
          _stream(stream) {}
    void visit(const BufferUploadCommand *cmd) noexcept override;
    void visit(const BufferDownloadCommand *cmd) noexcept override;
    void visit(const ImageUploadCommand *cmd) noexcept override;
    void visit(const ImageDownloadCommand *cmd) noexcept override;
    void visit(const MeshBuildCommand *cmd) noexcept override;
    void visit(const AccelBuildCommand *cmd) noexcept override;
    void visit(const SynchronizeCommand *cmd) noexcept override;
    void visit(const ShaderDispatchCommand *cmd) noexcept override;
};

}// namespace ocarina