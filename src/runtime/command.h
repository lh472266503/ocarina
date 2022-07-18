//
// Created by Zero on 06/07/2022.
//

#pragma once

#include "core/stl.h"
#include "resource.h"
#include "core/pool.h"

namespace ocarina {

#define OC_RUNTIME_CMD         \
    BufferUploadCommand,       \
        BufferDownloadCommand, \
        SynchronizeCommand,    \
        ShaderDispatchCommand

/// forward declare
#define OC_MAKE_CMD_FWD_DECL(CMD) class CMD;
#define OC_CMD_FWD_DECL MAP(OC_MAKE_CMD_FWD_DECL, OC_RUNTIME_CMD)
OC_CMD_FWD_DECL

/// declare command pool function
#define OC_COMMAND_POOL_FUNC(CMD) CMD##_pool()
#define OC_MAKE_COMMAND_POOL_FUNC_DECL(CMD) [[nodiscard]] Pool<CMD> &OC_COMMAND_POOL_FUNC(CMD) noexcept;
#define OC_MAKE_COMMAND_POOL_FUNCS_DECL MAP(OC_MAKE_COMMAND_POOL_FUNC_DECL, OC_RUNTIME_CMD)
OC_MAKE_COMMAND_POOL_FUNCS_DECL

/// implement command pool function
#define OC_COMMAND_POOL_FUNCTION_IMPL(CMD)          \
    Pool<CMD> &OC_COMMAND_POOL_FUNC(CMD) noexcept { \
        static Pool<CMD> pool;                      \
        return pool;                                \
    }
#define OC_COMMAND_POOL_FUNCTIONS_IMPL MAP(OC_COMMAND_POOL_FUNCTION_IMPL, OC_RUNTIME_CMD)

#define OC_MAKE_CMD_VISITOR_ACCEPT(CMD) \
    void accept(CommandVisitor &visitor) const noexcept override { visitor.visit(this); }
#define OC_MAKE_RECYCLE_FUNC(CMD) \
    void recycle() noexcept override { OC_COMMAND_POOL_FUNC(CMD).recycle(this); }
#define OC_MAKE_CMD_CREATOR(CMD)                                      \
    template<typename... Args>                                        \
    static CMD *create(Args &&...args) noexcept {                     \
        return OC_COMMAND_POOL_FUNC(CMD).create(OC_FORWARD(args)...); \
    }
#define OC_MAKE_CMD_CLONE_FUNC(CMD) \
    CMD *clone() noexcept { return CMD::create(*this); }

#define OC_MAKE_CMD_COMMON_FUNC(CMD) \
    OC_MAKE_RECYCLE_FUNC(CMD)        \
    OC_MAKE_CMD_VISITOR_ACCEPT(CMD)  \
    OC_MAKE_CMD_CREATOR(CMD)         \
    OC_MAKE_CMD_CLONE_FUNC(CMD)

class CommandVisitor {
public:
#define OC_MAKE_COMMAND_VISIT(CMD) virtual void visit(const CMD *cmd) noexcept = 0;
    MAP(OC_MAKE_COMMAND_VISIT, OC_RUNTIME_CMD)
};

class Command {
public:
    virtual ~Command() noexcept = default;
    virtual void accept(CommandVisitor &visitor) const noexcept = 0;
    virtual void recycle() noexcept = 0;
};

class BufferUploadCommand final : public Command {
private:
    const void *_host_ptr{};
    ptr_t _device_ptr{};
    size_t _size_in_bytes{};

public:
    BufferUploadCommand(const void *hp, ptr_t dp, size_t size)
        : _host_ptr(hp), _device_ptr(dp), _size_in_bytes(size) {}
    [[nodiscard]] const void *host_ptr() const noexcept { return _host_ptr; }
    [[nodiscard]] ptr_t device_ptr() const noexcept { return _device_ptr; }
    [[nodiscard]] size_t size_in_bytes() const noexcept { return _size_in_bytes; }
    OC_MAKE_CMD_COMMON_FUNC(BufferUploadCommand)
};

class BufferDownloadCommand final : public Command {
private:
    void *_host_ptr{};
    ptr_t _device_ptr{};
    size_t _size_in_bytes{};

public:
    BufferDownloadCommand(void *hp, ptr_t dp, size_t size)
        : _host_ptr(hp), _device_ptr(dp), _size_in_bytes(size) {}
    [[nodiscard]] void *host_ptr() const noexcept { return _host_ptr; }
    [[nodiscard]] ptr_t device_ptr() const noexcept { return _device_ptr; }
    [[nodiscard]] size_t size_in_bytes() const noexcept { return _size_in_bytes; }
    OC_MAKE_CMD_COMMON_FUNC(BufferDownloadCommand)
};

class SynchronizeCommand final : public Command {
public:
    OC_MAKE_CMD_COMMON_FUNC(SynchronizeCommand)
};

[[nodiscard]] inline SynchronizeCommand *synchronize() noexcept {
    return SynchronizeCommand::create();
}

class ShaderDispatchCommand final : public Command {
public:
    OC_MAKE_CMD_COMMON_FUNC(ShaderDispatchCommand)
};

}// namespace ocarina