//
// Created by zero on 2022/7/9.
//

#pragma once

#include "runtime/command.h"

namespace ocarina {

class CUDACommandVisitor final : public CommandVisitor {
    void visit(const BufferUploadCommand *cmd) noexcept override;
    void visit(const BufferDownloadCommand *cmd) noexcept override;
};


}