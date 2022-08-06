//
// Created by Zero on 06/07/2022.
//

#pragma once

#include "core/stl.h"
#include "resource.h"
#include "core/pool.h"
#include "core/image_base.h"

namespace ocarina {

#define OC_RUNTIME_CMD          \
    BufferUploadCommand,        \
        BufferDownloadCommand,  \
        TextureUploadCommand,   \
        TextureDownloadCommand, \
        SynchronizeCommand,     \
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

class DataOpCommand : public Command {
private:
    handle_ty _host_ptr{};
    handle_ty _device_ptr{};
    bool _async{};

protected:
    DataOpCommand(handle_ty hp, handle_ty dp, bool async)
        : _host_ptr(hp), _device_ptr(dp), _async(async) {}

public:
    template<typename T = handle_ty>
    [[nodiscard]] T host_ptr() const noexcept { return reinterpret_cast<T>(_host_ptr); }
    [[nodiscard]] bool async() const noexcept { return _async; }
    [[nodiscard]] handle_ty device_ptr() const noexcept { return _device_ptr; }
};

class BufferOpCommand : public DataOpCommand {
private:
    handle_ty _host_ptr{};
    handle_ty _device_ptr{};
    size_t _size_in_bytes{};
    bool _async{};

protected:
    BufferOpCommand(handle_ty hp, handle_ty dp, size_t size, bool async)
        : DataOpCommand(hp, dp, async), _size_in_bytes(size) {}

public:
    [[nodiscard]] size_t size_in_bytes() const noexcept { return _size_in_bytes; }
};

class BufferUploadCommand final : public BufferOpCommand {
public:
    BufferUploadCommand(const void *hp, handle_ty dp, size_t size, bool async = true)
        : BufferOpCommand(reinterpret_cast<handle_ty>(hp), dp, size, async) {}
    OC_MAKE_CMD_COMMON_FUNC(BufferUploadCommand)
};

class BufferDownloadCommand final : public BufferOpCommand {
public:
    BufferDownloadCommand(void *hp, handle_ty dp, size_t size, bool async = true)
        : BufferOpCommand(reinterpret_cast<handle_ty>(hp), dp, size, async) {}
    OC_MAKE_CMD_COMMON_FUNC(BufferDownloadCommand)
};

class TextureOpCommand : public DataOpCommand {
private:
    handle_ty _host_ptr{};
    handle_ty _device_ptr{};
    PixelStorage _pixel_storage{};
    uint2 _resolution{};
    bool _async{};

public:
    TextureOpCommand(handle_ty data, handle_ty device_ptr, uint2 resolution, PixelStorage storage, bool async)
        : DataOpCommand(data, device_ptr, async), _pixel_storage(storage), _resolution(resolution) {}
    [[nodiscard]] PixelStorage pixel_storage() const noexcept { return _pixel_storage; }
    [[nodiscard]] uint2 resolution() const noexcept { return _resolution; }
};

class TextureUploadCommand final : public TextureOpCommand {
public:
    TextureUploadCommand(const void *data, handle_ty device_ptr, uint2 resolution, PixelStorage storage, bool async)
        : TextureOpCommand(reinterpret_cast<handle_ty>(data), device_ptr, resolution, storage, async) {}
    OC_MAKE_CMD_COMMON_FUNC(TextureUploadCommand)
};

class TextureDownloadCommand final : public TextureOpCommand {
public:
    TextureDownloadCommand(void *data, handle_ty device_ptr, uint2 resolution, PixelStorage storage, bool async)
        : TextureOpCommand(reinterpret_cast<handle_ty>(data), device_ptr, resolution, storage, async) {}
    OC_MAKE_CMD_COMMON_FUNC(TextureDownloadCommand)
};

class SynchronizeCommand final : public Command {
public:
    OC_MAKE_CMD_COMMON_FUNC(SynchronizeCommand)
};

[[nodiscard]] inline SynchronizeCommand *synchronize() noexcept {
    return SynchronizeCommand::create();
}
class Function;
class ShaderDispatchCommand final : public Command {
private:
    const Function &_function;
    span<void *> _args;
    uint3 _dispatch_dim;
    handle_ty _entry{};

public:
    explicit ShaderDispatchCommand(const Function &function, handle_ty entry, span<void *> args, uint3 dim)
        : _function(function), _entry(entry), _args(args), _dispatch_dim(dim) {}
    [[nodiscard]] span<void *> args() noexcept { return _args; }
    [[nodiscard]] uint3 dispatch_dim() const noexcept { return _dispatch_dim; }
    [[nodiscard]] const Function &function() const noexcept { return _function; }
    [[nodiscard]] Function &function_nc() noexcept {
        return *const_cast<Function *>(&_function);
    }
    template<typename T>
    [[nodiscard]] auto entry() const noexcept { return reinterpret_cast<T>(_entry); }
    OC_MAKE_CMD_COMMON_FUNC(ShaderDispatchCommand)
};

}// namespace ocarina