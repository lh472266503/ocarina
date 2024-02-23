//
// Created by Zero on 2024/2/22.
//

#pragma once

#include "resource.h"
#include "rhi/command.h"
#include "rhi/stats.h"
#include "bindless_array.h"

namespace ocarina {

class ByteBufferView {
private:
    handle_ty _handle{};
    size_t _offset{};
    size_t _size{};
    size_t _total_size{};

public:
    ByteBufferView() = default;

    ByteBufferView(handle_ty handle, size_t offset, size_t size, size_t total_size)
        : _handle(handle), _offset(offset), _size(size), _total_size(total_size) {}

    [[nodiscard]] ByteBufferView subview(size_t offset, size_t size) const noexcept {
        return ByteBufferView(_handle, _offset + offset, size, _total_size);
    }

    [[nodiscard]] size_t size() const { return _size; }
    [[nodiscard]] size_t size_in_byte() const noexcept { return _size; }
    [[nodiscard]] size_t element_size() const noexcept { return 1; }
    [[nodiscard]] handle_ty head() const { return _handle + _offset * element_size(); }

    [[nodiscard]] BufferCopyCommand *copy_from(const ByteBufferView &src, bool async = true,
                                               uint dst_offset = 0) noexcept {
        return BufferCopyCommand::create(src.head(), head(), 0, dst_offset * element_size(),
                                         src.size_in_byte(), true);
    }

    template<typename Arg>
    requires is_buffer_or_view_v<Arg>
    [[nodiscard]] BufferCopyCommand *copy_to(Arg &dst, uint src_offset = 0,
                                             bool async = true) noexcept {
        return BufferCopyCommand::create(head(), dst.head(), src_offset * element_size(),
                                         0, dst.size_in_byte(), true);
    }

    [[nodiscard]] BufferDownloadCommand *download(void *data, uint src_offset = 0,
                                                  bool async = true) const noexcept {
        return BufferDownloadCommand::create(data, head() + src_offset * element_size(),
                                             size_in_byte(), async);
    }

    [[nodiscard]] BufferUploadCommand *upload(const void *data, bool async = true) const noexcept {
        return BufferUploadCommand::create(data, head(), size_in_byte(), async);
    }

    [[nodiscard]] BufferByteSetCommand *byte_set(uchar value, bool async = true) const noexcept {
        return BufferByteSetCommand::create(head(), size_in_byte(), value, async);
    }

    [[nodiscard]] BufferByteSetCommand *reset(bool async = true) const noexcept {
        return byte_set(0, async);
    }
};

class ByteBuffer : public RHIResource {
private:
    /// just for construct memory block
    mutable BufferProxy _proxy{};
    size_t _size{};

public:
    ByteBuffer(Device::Impl *device, size_t size, const string &desc = "")
        : RHIResource(device, Tag::BUFFER, device->create_buffer(size, desc)),
          _size(size) {}

    void destroy() override {
        _destroy();
        _size = 0;
    }

    [[nodiscard]] const void *proxy_ptr() const noexcept {
        _proxy.handle = _handle;
        _proxy.size = _size;
        return &_proxy;
    }

    [[nodiscard]] size_t data_alignment() const noexcept override {
        return alignof(decltype(_proxy));
    }

    [[nodiscard]] size_t data_size() const noexcept override {
        return sizeof(_proxy);
    }

    [[nodiscard]] MemoryBlock memory_block() const noexcept override {
        return {proxy_ptr(), data_size(), data_alignment(), max_member_size()};
    }

    template<typename U = uint>
    [[nodiscard]] const Expression *expression() const noexcept {
        const CapturedResource &captured_resource = Function::current()->get_captured_resource(Type::of<decltype(*this)>(),
                                                                                               Variable::Tag::BUFFER,
                                                                                               memory_block());
        return captured_resource.expression();
    }

    [[nodiscard]] ByteBufferView view(size_t offset = 0, size_t size = 0) const noexcept {
        size = size == 0 ? _size - offset : size;
        return ByteBufferView(_handle, offset, size, _size);
    }
    template<typename... Args>
    [[nodiscard]] BufferCopyCommand *copy_from(Args &&...args) const noexcept {
        return view(0, _size).copy_from(OC_FORWARD(args)...);
    }
    template<typename... Args>
    [[nodiscard]] BufferCopyCommand *download(Args &&...args) const noexcept {
        return view(0, _size).download(OC_FORWARD(args)...);
    }
    template<typename... Args>
    [[nodiscard]] BufferCopyCommand *upload(Args &&...args) const noexcept {
        return view(0, _size).upload(OC_FORWARD(args)...);
    }
    template<typename... Args>
    [[nodiscard]] BufferCopyCommand *byte_set(Args &&...args) const noexcept {
        return view(0, _size).byte_set(OC_FORWARD(args)...);
    }
    template<typename... Args>
    [[nodiscard]] BufferCopyCommand *reset(Args &&...args) const noexcept {
        return view(0, _size).reset(OC_FORWARD(args)...);
    }
};
}// namespace ocarina