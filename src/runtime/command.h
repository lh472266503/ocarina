//
// Created by Zero on 06/07/2022.
//

#pragma once

#include "core/stl.h"
#include "resource.h"

namespace ocarina {

class BufferUploadCommand;
class BufferDownloadCommand;

class CommandVisitor {
public:
    virtual void visit(const BufferUploadCommand *cmd) noexcept = 0;
    virtual void visit(const BufferDownloadCommand *cmd) noexcept = 0;
};

class Command {
public:
    virtual ~Command() noexcept = default;
    virtual void accept(CommandVisitor &visitor) const noexcept = 0;
};

#define OC_MAKE_CMD_VISITOR_ACCEPT \
    void accept(CommandVisitor &visitor) const noexcept override { visitor.visit(this); }

class BufferUploadCommand final : public Command {
private:
    const void *_host_ptr{};
    size_t _offset{};
    ptr_t _device_ptr{};
    size_t _size_in_bytes{};

public:
    OC_MAKE_CMD_VISITOR_ACCEPT
};

class BufferDownloadCommand final : public Command {
private:
    void *_host_ptr{};
    size_t _offset{};
    ptr_t _device_ptr{};
    size_t _size_in_bytes{};

public:
    OC_MAKE_CMD_VISITOR_ACCEPT
};

}// namespace ocarina