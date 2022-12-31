//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "resource.h"
#include "rhi/command.h"

namespace ocarina {

template<typename T, int... Dims>
class Buffer;

template<typename T, int... Dims>
class BufferView {
public:
    static constexpr size_t element_size = sizeof(T);

private:
    handle_ty _handle{};
    size_t _offset{};
    size_t _size{};
    size_t _total_size{};

public:
    BufferView(const Buffer<T, Dims...> &buffer);
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

template<typename T = std::byte, int... Dims>
class Buffer : public RHIResource {
public:
    static constexpr size_t element_size = sizeof(T);
    using element_type = T;
    static constexpr std::array<int, sizeof...(Dims)> dims = {Dims...};
    static constexpr bool has_multi_dim() noexcept { return !dims.empty(); }

private:
    size_t _size{};

public:
    Buffer() = default;

    Buffer(Device::Impl *device, size_t size)
        : RHIResource(device, Tag::BUFFER, device->create_buffer(size * sizeof(T))),
          _size(size) {}

    Buffer(BufferView<T, Dims...> buffer_view)
        : RHIResource(nullptr, Tag::BUFFER, buffer_view.head()),
          _size(buffer_view.size()) {}

    [[nodiscard]] BufferView<T> view(size_t offset = 0, size_t size = 0) const noexcept {
        size = size == 0 ? _size - offset : size;
        return BufferView<T>(_handle, offset, size, _size);
    }

    // Move constructor
    Buffer(Buffer &&other) noexcept
        : RHIResource(std::move(other)) {
        this->_size = other._size;
    }

    // Move assignment
    Buffer &operator=(Buffer &&other) noexcept {
        RHIResource::operator=(std::move(other));
        this->_size = other._size;
        return *this;
    }

    template<typename U>
    [[nodiscard]] auto ptr() const noexcept {
        if constexpr (std::is_same_v<U, handle_ty>) {
            return handle();
        } else {
            return reinterpret_cast<U>(handle());
        }
    }

    template<typename U = void *>
    [[nodiscard]] auto address(size_t offset) const noexcept {
        return (U) (handle() + offset * element_size);
    }

    /// head of the buffer
    [[nodiscard]] handle_ty head() const noexcept { return handle(); }

    /// for dsl trait
    auto operator[](int i) { return T{}; }

    template<typename ...Index>
    requires concepts::all_integral<expr_value_t<Index>...>
        OC_NODISCARD auto
        read(Index &&...index) const {
        const ArgumentBinding &uniform = Function::current()->get_uniform_var(Type::of<decltype(*this)>(),
                                                                              Variable::Tag::BUFFER,
                                                                              memory_block());
        return make_expr<Buffer<T>>(uniform.expression()).read(OC_FORWARD(index)...);
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<element_type, expr_value_t<Val>>
    void write(Index &&index, Val &&elm) {
        const ArgumentBinding &uniform = Function::current()->get_uniform_var(Type::of<decltype(*this)>(),
                                                                              Variable::Tag::BUFFER,
                                                                              memory_block());
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

    void upload_immediately(const void *data) const noexcept {
        upload_sync(data)->accept(*_device->command_visitor());
    }

    void upload_immediately(const void *data, size_t offset, size_t size) const noexcept {
        view(offset, size).upload_sync(data)->accept(*_device->command_visitor());
    }

    void download_immediately(void *data) const noexcept {
        download_sync(data)->accept(*_device->command_visitor());
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

template<typename T, int... dims>
BufferView<T, dims...>::BufferView(const Buffer<T, dims...> &buffer)
    : BufferView(buffer.handle(), buffer.size()) {}

}// namespace ocarina