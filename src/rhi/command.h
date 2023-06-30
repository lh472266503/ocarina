//
// Created by Zero on 06/07/2022.
//

#pragma once

#include "core/stl.h"
#include "resources/resource.h"
#include "core/pool.h"
#include "core/image_base.h"

namespace ocarina {

#define OC_RUNTIME_CMD              \
    BufferUploadCommand,            \
        BufferDownloadCommand,      \
        BufferByteSetCommand,       \
        BufferCopyCommand,          \
        BufferToTextureCommand,     \
        TextureUploadCommand,       \
        TextureDownloadCommand,     \
        TextureCopyCommand,         \
        HostFunctionCommand,        \
        SynchronizeCommand,         \
        BLASBuildCommand,           \
        TLASBuildCommand,           \
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

    virtual ~CommandVisitor() = default;
};

class Command {
private:
    bool _async{};

public:
    explicit Command(bool async = true) : _async(async) {}
    virtual ~Command() noexcept = default;
    virtual void accept(CommandVisitor &visitor) const noexcept = 0;
    virtual void recycle() noexcept = 0;
    [[nodiscard]] bool async() const noexcept { return _async; }
};

class BufferCommand : public Command {
protected:
    handle_ty _device_ptr{};
    size_t _size_in_bytes{};

public:
    BufferCommand(handle_ty dp, size_t size, bool async = true)
        : Command(async), _device_ptr(dp), _size_in_bytes(size) {}
    [[nodiscard]] size_t size_in_bytes() const noexcept { return _size_in_bytes; }
    [[nodiscard]] handle_ty device_ptr() const noexcept { return _device_ptr; }
};

class DataCopyCommand : public Command {
protected:
    handle_ty _src{};
    handle_ty _dst{};

public:
    DataCopyCommand(handle_ty src, handle_ty dst, bool async)
        : Command(async), _src(src), _dst(dst) {}
    template<typename T = handle_ty>
    [[nodiscard]] T src() const noexcept {
        if constexpr (std::is_same_v<std::remove_cvref_t<T>, handle_ty>) {
            return _src;
        } else {
            return reinterpret_cast<T>(_src);
        }
    }
    template<typename T = handle_ty>
    [[nodiscard]] T dst() const noexcept {
        if constexpr (std::is_same_v<std::remove_cvref_t<T>, handle_ty>) {
            return _dst;
        } else {
            return reinterpret_cast<T>(_dst);
        }
    }
};

class BufferCopyCommand : public DataCopyCommand {
private:
    size_t _src_offset;
    size_t _dst_offset;
    size_t _size;

public:
    BufferCopyCommand(uint64_t src, uint64_t dst, size_t src_offset, size_t dst_offset, size_t size, bool async) noexcept
        : DataCopyCommand{src, dst, async},
          _src_offset{src_offset}, _dst_offset{dst_offset}, _size{size} {}
    [[nodiscard]] size_t src_offset() const noexcept { return _src_offset; }
    [[nodiscard]] size_t dst_offset() const noexcept { return _dst_offset; }
    [[nodiscard]] size_t size() const noexcept { return _size; }
    OC_MAKE_CMD_COMMON_FUNC(BufferCopyCommand)
};

class TextureCopyCommand : public DataCopyCommand {
private:
    PixelStorage _storage;
    uint3 _res;
    uint _src_level;
    uint _dst_level;

public:
    TextureCopyCommand(uint64_t src, uint64_t dst, uint3 res, PixelStorage pixel_storage,
                       uint src_level, uint dst_level, bool async) noexcept
        : DataCopyCommand{src, dst, async},
          _res(res), _src_level(src_level),
          _dst_level(dst_level),
          _storage(pixel_storage) {}

    [[nodiscard]] uint src_level() const noexcept { return _src_level; }
    [[nodiscard]] uint dst_level() const noexcept { return _dst_level; }
    [[nodiscard]] PixelStorage pixel_storage() const noexcept { return _storage; }
    [[nodiscard]] uint3 resolution() const noexcept { return _res; }
    OC_MAKE_CMD_COMMON_FUNC(TextureCopyCommand)
};

class DataOpCommand : public Command {
protected:
    handle_ty _host_ptr{};
    handle_ty _device_ptr{};

protected:
    DataOpCommand(handle_ty hp, handle_ty dp, bool async)
        : Command(async), _host_ptr(hp), _device_ptr(dp) {}

public:
    template<typename T = handle_ty>
    [[nodiscard]] T host_ptr() const noexcept { return reinterpret_cast<T>(_host_ptr); }
    [[nodiscard]] handle_ty host_ptr() const noexcept { return _host_ptr; }
    template<typename T>
    [[nodiscard]] T device_ptr() const noexcept { return reinterpret_cast<T>(_device_ptr); }
    [[nodiscard]] handle_ty device_ptr() const noexcept { return _device_ptr; }
};

class BufferOpCommand : public DataOpCommand {
private:
    size_t _size_in_bytes{};

protected:
    BufferOpCommand(handle_ty hp, handle_ty dp, size_t size, bool async)
        : DataOpCommand(hp, dp, async), _size_in_bytes(size) {}

public:
    [[nodiscard]] size_t size_in_bytes() const noexcept { return _size_in_bytes; }
};

class BufferByteSetCommand final : public BufferCommand {
public:
    uchar _val{};

public:
    BufferByteSetCommand(handle_ty dp, size_t size, uchar val = 0, bool async = true)
        : BufferCommand(dp, size, async), _val(val) {}
    [[nodiscard]] uchar value() const noexcept { return _val; }
    OC_MAKE_CMD_COMMON_FUNC(BufferByteSetCommand)
};

class BufferUploadCommand final : public BufferOpCommand {
public:
    BufferUploadCommand(const void *hp, handle_ty dp, size_t size, bool async = true)
        : BufferOpCommand(reinterpret_cast<handle_ty>(hp), dp, size, async) {}
    OC_MAKE_CMD_COMMON_FUNC(BufferUploadCommand)
};

class BufferToTextureCommand final : public DataCopyCommand {
private:
    PixelStorage _storage;
    size_t _buffer_offset;
    uint3 _res;
    uint _level;

public:
    BufferToTextureCommand(handle_ty src, size_t buffer_offset,
                           handle_ty dst, PixelStorage ps,
                           uint3 res, size_t level, bool async)
        : DataCopyCommand(src, dst, async), _storage(ps), _res(res),
          _buffer_offset(buffer_offset), _level(level) {}
    [[nodiscard]] PixelStorage pixel_storage() const noexcept { return _storage; }
    [[nodiscard]] size_t buffer_offset() const noexcept { return _buffer_offset; }
    [[nodiscard]] size_t width() const noexcept { return _res.x; }
    [[nodiscard]] size_t height() const noexcept { return _res.y; }
    [[nodiscard]] size_t depth() const noexcept { return _res.z; }
    [[nodiscard]] size_t width_in_bytes() const noexcept { return pixel_size(_storage) * width(); }
    [[nodiscard]] size_t size_in_bytes() const noexcept { return height() * width_in_bytes(); }
    [[nodiscard]] uint level() const noexcept { return _level; }
    [[nodiscard]] uint3 resolution() const noexcept { return _res; }
    OC_MAKE_CMD_COMMON_FUNC(BufferToTextureCommand)
};

class BufferDownloadCommand final : public BufferOpCommand {
public:
    BufferDownloadCommand(void *hp, handle_ty dp, size_t size, bool async = true)
        : BufferOpCommand(reinterpret_cast<handle_ty>(hp), dp, size, async) {}
    OC_MAKE_CMD_COMMON_FUNC(BufferDownloadCommand)
};

class TextureOpCommand : public DataOpCommand {
private:
    PixelStorage _pixel_storage{};
    uint3 _resolution{};

public:
    TextureOpCommand(handle_ty data, handle_ty device_ptr, uint2 resolution, PixelStorage storage, bool async)
        : DataOpCommand(data, device_ptr, async), _pixel_storage(storage), _resolution(resolution.x, resolution.y, 1) {}
    TextureOpCommand(handle_ty data, handle_ty device_ptr, uint3 resolution, PixelStorage storage, bool async)
        : DataOpCommand(data, device_ptr, async), _pixel_storage(storage), _resolution(resolution) {}
    [[nodiscard]] PixelStorage pixel_storage() const noexcept { return _pixel_storage; }
    [[nodiscard]] size_t width() const noexcept { return _resolution.x; }
    [[nodiscard]] size_t height() const noexcept { return _resolution.y; }
    [[nodiscard]] size_t depth() const noexcept { return _resolution.z; }
    [[nodiscard]] size_t width_in_bytes() const noexcept { return pixel_size(_pixel_storage) * width(); }
    [[nodiscard]] size_t size_in_bytes() const noexcept { return height() * width_in_bytes(); }
    [[nodiscard]] uint3 resolution() const noexcept { return _resolution; }
};

class TextureUploadCommand final : public TextureOpCommand {
public:
    TextureUploadCommand(const void *data, handle_ty device_ptr, uint3 resolution, PixelStorage storage, bool async)
        : TextureOpCommand(reinterpret_cast<handle_ty>(data), device_ptr, resolution, storage, async) {}
    OC_MAKE_CMD_COMMON_FUNC(TextureUploadCommand)
};

class TextureDownloadCommand final : public TextureOpCommand {
public:
    TextureDownloadCommand(void *data, handle_ty device_ptr, uint3 resolution, PixelStorage storage, bool async)
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

class BLASBuildCommand final : public Command {
private:
    handle_ty _mesh{};

public:
    explicit BLASBuildCommand(handle_ty mesh) : _mesh(mesh) {}
    template<typename T>
    [[nodiscard]] T *mesh() const { return reinterpret_cast<T *>(_mesh); }
    OC_MAKE_CMD_COMMON_FUNC(BLASBuildCommand)
};

class TLASBuildCommand final : public Command {
private:
    handle_ty _accel{};

public:
    explicit TLASBuildCommand(handle_ty accel) : _accel(accel) {}
    template<typename T>
    [[nodiscard]] T *accel() const { return reinterpret_cast<T *>(_accel); }
    OC_MAKE_CMD_COMMON_FUNC(TLASBuildCommand)
};

class Function;
class ArgumentList;
class ShaderDispatchCommand final : public Command {
private:
    SP<ArgumentList> _argument_list;
    uint3 _dispatch_dim;
    handle_ty _entry{};

public:
    ShaderDispatchCommand(handle_ty entry, SP<ArgumentList> argument_list, uint3 dim);
    [[nodiscard]] span<void *> args() noexcept;
    [[nodiscard]] span<const MemoryBlock> params() noexcept;
    [[nodiscard]] size_t params_size() noexcept;
    [[nodiscard]] uint3 dispatch_dim() const noexcept { return _dispatch_dim; }

    template<typename T>
    [[nodiscard]] auto entry() const noexcept { return reinterpret_cast<T>(_entry); }
    OC_MAKE_CMD_COMMON_FUNC(ShaderDispatchCommand)
};

class HostFunctionCommand : public Command {
private:
    std::function<void()> _function;

public:
    HostFunctionCommand(std::function<void()> f, bool async)
        : Command(async), _function(ocarina::move(f)) {}
    [[nodiscard]] std::function<void()> function() const noexcept { return _function; }
    OC_MAKE_CMD_COMMON_FUNC(HostFunctionCommand)
};

}// namespace ocarina