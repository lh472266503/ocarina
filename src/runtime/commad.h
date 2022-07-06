//
// Created by Zero on 06/07/2022.
//

#pragma once

#include "core/stl.h"

namespace ocarina {

class BufferUploadCommand;
class BufferDownloadCommand;

class CommandVisitor;

class Command {
public:
    virtual ~Command() noexcept = default;
    virtual void accept(CommandVisitor &visitor) const noexcept = 0;
    [[nodiscard]] virtual Command *clone() const noexcept = 0;
};

class BufferUploadCommand final : public Command {

};

}