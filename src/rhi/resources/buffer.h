//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "resource.h"
#include "rhi/command.h"

namespace ocarina {

template<typename T>
class Buffer;

template<typename T>
class BufferView {
public:
    static constexpr size_t element_size = sizeof(T);

private:
    handle_ty _handle{};
    size_t _offset{};
    size_t _size{};
    size_t _total_size{};

public:
    BufferView(const Buffer<T> &buffer);
    [[nodiscard]] handle_ty handle() const { return _handle; }
    [[nodiscard]] size_t size() const { return _size; }
    BufferView(handle_ty handle, size_t offset, size_t size, size_t total_size)
        : _handle(handle), _offset(offset), _size(size), _total_size(total_size) {}

    BufferView(handle_ty handle, size_t total_size)
        : _handle(handle), _offset(0), _total_size(total_size), _size(total_size) {}

    [[nodiscard]] handle_ty head() const { return _handle + _offset * element_size; }

    [[nodiscard]] BufferView<T> subview(size_t offset, size_t size) const noexcept {
        return BufferView<T>(_handle, _offset + offset, size, _total_size);
    }

    [[nodiscard]] BufferUploadCommand *upload(const void *data) const noexcept {
        return BufferUploadCommand::create(data, head(), _size * element_size, true);
    }

    [[nodiscard]] BufferUploadCommand *upload_sync(const void *data) const noexcept {
        return BufferUploadCommand::create(data, head(), _size * element_size, false);
    }

    [[nodiscard]] BufferDownloadCommand *download(void *data) const noexcept {
        return BufferDownloadCommand::create(data, head(), _size * element_size, true);
    }

    [[nodiscard]] BufferDownloadCommand *download_sync(void *data) const noexcept {
        return BufferDownloadCommand::create(data, head(), _size * element_size, false);
    }
};

template<typename T = std::byte>
class Buffer : public RHIResource {
public:
    static constexpr size_t element_size = sizeof(T);
    using element_type = T;

private:
    size_t _size{};

public:
    Buffer(Device::Impl *device, size_t size)
        : RHIResource(device, Tag::BUFFER, device->create_buffer(size * sizeof(T))),
          _size(size) {}

    Buffer(BufferView<T> buffer_view)
        : RHIResource(nullptr, Tag::BUFFER, buffer_view.head()),
          _size(buffer_view.size()) {}

    [[nodiscard]] BufferView<T> view(size_t offset = 0, size_t size = 0) const noexcept {
        size = size == 0 ? _size : size;
        return BufferView<T>(_handle, offset, size, _size);
    }

    /// head of the buffer
    [[nodiscard]] handle_ty head() const noexcept { return handle(); }

    /// for dsl trait
    auto operator[](int i) { return T{}; }

    template<typename Index>
    requires ocarina::is_integral_v<expr_value_t<Index>>
    OC_NODISCARD auto
    read(Index &&index) const {
        const UniformBinding &uniform = Function::current()->get_uniform_var(Type::of<Buffer<T>>(),
                                                                             handle_ptr(), Variable::Tag::BUFFER);
        return make_expr<Buffer<T>>(uniform.expression()).read(OC_FORWARD(index));
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<element_type, expr_value_t<Val>>
    void write(Index &&index, Val &&elm) {
        const UniformBinding &uniform = Function::current()->get_uniform_var(Type::of<Buffer<T>>(),
                                                                             handle_ptr(),
                                                                             Variable::Tag::BUFFER);
        const AccessExpr *expr = Function::current()->access(Type::of<element_type>(),
                                                             uniform.expression(),
                                                             OC_EXPR(index));
        assign(expr, OC_FORWARD(elm));
    }

    [[nodiscard]] size_t size() const noexcept { return _size; }

    template<typename... Args>
    [[nodiscard]] BufferUploadCommand *upload(Args &&...args) const noexcept {
        return view(0, _size).upload(OC_FORWARD(args)...);
    }

    template<typename... Args>
    [[nodiscard]] BufferDownloadCommand *download(Args &&...args) const noexcept {
        return view(0, _size).download(OC_FORWARD(args)...);
    }

    template<typename... Args>
    [[nodiscard]] BufferUploadCommand *upload_sync(Args &&...args) const noexcept {
        return view(0, _size).upload_sync(OC_FORWARD(args)...);
    }

    template<typename... Args>
    [[nodiscard]] BufferDownloadCommand *download_sync(Args &&...args) const noexcept {
        return view(0, _size).download_sync(OC_FORWARD(args)...);
    }
};

template<typename T>
BufferView<T>::BufferView(const Buffer<T> &buffer)
    : BufferView(buffer.handle(), buffer.size()) {}

}// namespace ocarina