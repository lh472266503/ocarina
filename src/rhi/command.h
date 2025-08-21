//
// Created by Zero on 06/07/2022.
//

#pragma once

#include "core/stl.h"
#include "resources/resource.h"
#include "core/pool.h"
#include "core/image_base.h"

namespace ocarina {

#define OC_RUNTIME_CMD           \
    BufferUploadCommand,         \
        BufferDownloadCommand,   \
        BufferByteSetCommand,    \
        BufferCopyCommand,       \
        BufferReallocateCommand, \
        BufferToTextureCommand,  \
        TextureUploadCommand,    \
        TextureDownloadCommand,  \
        TextureCopyCommand,      \
        HostFunctionCommand,     \
        SynchronizeCommand,      \
        BLASBuildCommand,        \
        TLASBuildCommand,        \
        TLASUpdateCommand,       \
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
    bool async_{};

public:
    explicit Command(bool async = true) : async_(async) {}
    virtual ~Command() noexcept = default;
    virtual void accept(CommandVisitor &visitor) const noexcept = 0;
    virtual void recycle() noexcept = 0;
    [[nodiscard]] bool async() const noexcept { return async_; }
};

class BufferCommand : public Command {
protected:
    handle_ty device_handle_{};
    size_t size_in_bytes_{};

public:
    BufferCommand(handle_ty dp, size_t size, bool async = true)
        : Command(async), device_handle_(dp), size_in_bytes_(size) {}
    [[nodiscard]] size_t size_in_bytes() const noexcept { return size_in_bytes_; }
    [[nodiscard]] handle_ty device_handle() const noexcept { return device_handle_; }
};

class DataCopyCommand : public Command {
protected:
    handle_ty src_{};
    handle_ty dst_{};

public:
    DataCopyCommand(handle_ty src, handle_ty dst, bool async)
        : Command(async), src_(src), dst_(dst) {}
    template<typename T = handle_ty>
    [[nodiscard]] T src() const noexcept {
        if constexpr (std::is_same_v<std::remove_cvref_t<T>, handle_ty>) {
            return src_;
        } else {
            return reinterpret_cast<T>(src_);
        }
    }
    template<typename T = handle_ty>
    [[nodiscard]] T dst() const noexcept {
        if constexpr (std::is_same_v<std::remove_cvref_t<T>, handle_ty>) {
            return dst_;
        } else {
            return reinterpret_cast<T>(dst_);
        }
    }
};

class BufferCopyCommand : public DataCopyCommand {
private:
    size_t src_offset_;
    size_t dst_offset_;
    size_t size_;

public:
    BufferCopyCommand(uint64_t src, uint64_t dst, size_t src_offset, size_t dst_offset, size_t size, bool async) noexcept
        : DataCopyCommand{src, dst, async},
          src_offset_{src_offset}, dst_offset_{dst_offset}, size_{size} {}
    [[nodiscard]] size_t src_offset() const noexcept { return src_offset_; }
    [[nodiscard]] size_t dst_offset() const noexcept { return dst_offset_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    OC_MAKE_CMD_COMMON_FUNC(BufferCopyCommand)
};

class TextureCopyCommand : public DataCopyCommand {
private:
    PixelStorage storage_;
    uint3 res_;
    uint src_level_;
    uint dst_level_;

public:
    TextureCopyCommand(uint64_t src, uint64_t dst, uint3 res, PixelStorage pixel_storage,
                       uint src_level, uint dst_level, bool async) noexcept
        : DataCopyCommand{src, dst, async},
          res_(res), src_level_(src_level),
          dst_level_(dst_level),
          storage_(pixel_storage) {}

    [[nodiscard]] uint src_level() const noexcept { return src_level_; }
    [[nodiscard]] uint dst_level() const noexcept { return dst_level_; }
    [[nodiscard]] PixelStorage pixel_storage() const noexcept { return storage_; }
    [[nodiscard]] uint3 resolution() const noexcept { return res_; }
    OC_MAKE_CMD_COMMON_FUNC(TextureCopyCommand)
};

class DataOpCommand : public Command {
protected:
    handle_ty host_ptr_{};
    handle_ty device_handle_{};
    size_t device_offset_{};

protected:
    DataOpCommand(handle_ty hp, handle_ty dp, size_t d_offset, bool async)
        : Command(async), host_ptr_(hp), device_offset_(d_offset), device_handle_(dp) {}

public:
    template<typename T = handle_ty>
    [[nodiscard]] T host_ptr() const noexcept { return reinterpret_cast<T>(host_ptr_); }
    [[nodiscard]] handle_ty host_ptr() const noexcept { return host_ptr_; }
    template<typename T>
    [[nodiscard]] T device_handle() const noexcept { return reinterpret_cast<T>(device_handle_); }
    [[nodiscard]] handle_ty device_handle() const noexcept { return device_handle_; }
    OC_MAKE_MEMBER_GETTER(device_offset, )
};

class BufferOpCommand : public DataOpCommand {
private:
    size_t size_in_bytes_{};

protected:
    BufferOpCommand(handle_ty hp, handle_ty dp, size_t offset, size_t size, bool async)
        : DataOpCommand(hp, dp, offset, async), size_in_bytes_(size) {}

public:
    [[nodiscard]] size_t size_in_bytes() const noexcept { return size_in_bytes_; }
};

class RHIResource;

class BufferReallocateCommand : public Command {
private:
    RHIResource *rhi_resource_{};
    size_t new_size_;

public:
    BufferReallocateCommand(RHIResource *rhi_resource, size_t new_size, bool async = true)
        : Command(async), new_size_(new_size), rhi_resource_(rhi_resource) {}
    [[nodiscard]] RHIResource *rhi_resource() const noexcept { return rhi_resource_; }
    [[nodiscard]] size_t new_size() const noexcept { return new_size_; }
    OC_MAKE_CMD_COMMON_FUNC(BufferReallocateCommand)
};

class BufferByteSetCommand final : public BufferCommand {
private:
    uchar val_{};

public:
    BufferByteSetCommand(handle_ty dp, size_t size, uchar val = 0, bool async = true)
        : BufferCommand(dp, size, async), val_(val) {}
    [[nodiscard]] uchar value() const noexcept { return val_; }
    OC_MAKE_CMD_COMMON_FUNC(BufferByteSetCommand)
};

class BufferUploadCommand final : public BufferOpCommand {
public:
    BufferUploadCommand(const void *hp, handle_ty dp, size_t d_offset, size_t size, bool async)
        : BufferOpCommand(reinterpret_cast<handle_ty>(hp), dp, d_offset, size, async) {}
    OC_MAKE_CMD_COMMON_FUNC(BufferUploadCommand)
};

class BufferToTextureCommand final : public DataCopyCommand {
private:
    PixelStorage storage_;
    size_t buffer_offset_;
    uint3 res_;
    uint level_;

public:
    BufferToTextureCommand(handle_ty src, size_t buffer_offset,
                           handle_ty dst, PixelStorage ps,
                           uint3 res, size_t level, bool async)
        : DataCopyCommand(src, dst, async), storage_(ps), res_(res),
          buffer_offset_(buffer_offset), level_(level) {}
    [[nodiscard]] PixelStorage pixel_storage() const noexcept { return storage_; }
    [[nodiscard]] size_t buffer_offset() const noexcept { return buffer_offset_; }
    [[nodiscard]] size_t width() const noexcept { return res_.x; }
    [[nodiscard]] size_t height() const noexcept { return res_.y; }
    [[nodiscard]] size_t depth() const noexcept { return res_.z; }
    [[nodiscard]] size_t width_in_bytes() const noexcept { return pixel_size(storage_) * width(); }
    [[nodiscard]] size_t size_in_bytes() const noexcept { return height() * width_in_bytes(); }
    [[nodiscard]] uint level() const noexcept { return level_; }
    [[nodiscard]] uint3 resolution() const noexcept { return res_; }
    OC_MAKE_CMD_COMMON_FUNC(BufferToTextureCommand)
};

class BufferDownloadCommand final : public BufferOpCommand {
public:
    BufferDownloadCommand(void *hp, handle_ty dp, size_t d_offset, size_t size, bool async)
        : BufferOpCommand(reinterpret_cast<handle_ty>(hp), dp, d_offset, size, async) {}
    OC_MAKE_CMD_COMMON_FUNC(BufferDownloadCommand)
};

class TextureOpCommand : public DataOpCommand {
private:
    PixelStorage pixel_storage_{};
    uint3 resolution_{};

public:
    TextureOpCommand(handle_ty data, handle_ty device_handle, uint2 resolution, PixelStorage storage, bool async)
        : DataOpCommand(data, device_handle, 0, async), pixel_storage_(storage), resolution_(resolution.x, resolution.y, 1) {}
    TextureOpCommand(handle_ty data, handle_ty device_handle, uint3 resolution, PixelStorage storage, bool async)
        : DataOpCommand(data, device_handle, 0, async), pixel_storage_(storage), resolution_(resolution) {}
    [[nodiscard]] PixelStorage pixel_storage() const noexcept { return pixel_storage_; }
    [[nodiscard]] size_t width() const noexcept { return resolution_.x; }
    [[nodiscard]] size_t height() const noexcept { return resolution_.y; }
    [[nodiscard]] size_t depth() const noexcept { return resolution_.z; }
    [[nodiscard]] size_t width_in_bytes() const noexcept { return pixel_size(pixel_storage_) * width(); }
    [[nodiscard]] size_t size_in_bytes() const noexcept { return height() * width_in_bytes(); }
    [[nodiscard]] uint3 resolution() const noexcept { return resolution_; }
};

class TextureUploadCommand final : public TextureOpCommand {
public:
    TextureUploadCommand(const void *data, handle_ty device_handle, uint3 resolution, PixelStorage storage, bool async)
        : TextureOpCommand(reinterpret_cast<handle_ty>(data), device_handle, resolution, storage, async) {}
    OC_MAKE_CMD_COMMON_FUNC(TextureUploadCommand)
};

class TextureDownloadCommand final : public TextureOpCommand {
public:
    TextureDownloadCommand(void *data, handle_ty device_handle, uint3 resolution, PixelStorage storage, bool async)
        : TextureOpCommand(reinterpret_cast<handle_ty>(data), device_handle, resolution, storage, async) {}
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
    handle_ty mesh_{};

public:
    explicit BLASBuildCommand(handle_ty mesh) : mesh_(mesh) {}
    template<typename T>
    [[nodiscard]] T *mesh() const { return reinterpret_cast<T *>(mesh_); }
    OC_MAKE_CMD_COMMON_FUNC(BLASBuildCommand)
};

class TLASBuildCommand final : public Command {
private:
    handle_ty accel_{};

public:
    explicit TLASBuildCommand(handle_ty accel) : accel_(accel) {}
    template<typename T>
    [[nodiscard]] T *accel() const { return reinterpret_cast<T *>(accel_); }
    OC_MAKE_CMD_COMMON_FUNC(TLASBuildCommand)
};

class TLASUpdateCommand final : public Command {
private:
    handle_ty accel_{};

public:
    explicit TLASUpdateCommand(handle_ty accel) : accel_(accel) {}
    template<typename T>
    [[nodiscard]] T *accel() const { return reinterpret_cast<T *>(accel_); }
    OC_MAKE_CMD_COMMON_FUNC(TLASUpdateCommand)
};

class Function;
class ArgumentList;
class ShaderDispatchCommand final : public Command {
private:
    SP<ArgumentList> argument_list_;
    uint3 dispatch_dim_;
    handle_ty entry_{};

public:
    ShaderDispatchCommand(handle_ty entry, SP<ArgumentList> argument_list, uint3 dim);
    [[nodiscard]] span<void *> args() noexcept;
    [[nodiscard]] span<const std::byte> argument_data() noexcept;
    [[nodiscard]] size_t params_size() noexcept;
    [[nodiscard]] uint3 dispatch_dim() const noexcept { return dispatch_dim_; }

    template<typename T>
    [[nodiscard]] auto entry() const noexcept { return reinterpret_cast<T>(entry_); }
    OC_MAKE_CMD_COMMON_FUNC(ShaderDispatchCommand)
};

class HostFunctionCommand : public Command {
private:
    std::function<void()> function_;

public:
    HostFunctionCommand(std::function<void()> f, bool async)
        : Command(async), function_(ocarina::move(f)) {}
    [[nodiscard]] std::function<void()> function() const noexcept { return function_; }
    OC_MAKE_CMD_COMMON_FUNC(HostFunctionCommand)
};

}// namespace ocarina