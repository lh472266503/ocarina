//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "resource.h"
#include "rhi/command.h"
#include "resource_array.h"

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
    [[nodiscard]] size_t size_in_byte() const noexcept { return _size * element_size; }
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

    [[nodiscard]] BufferDownloadCommand *download(void *data, uint src_offset = 0) const noexcept {
        return BufferDownloadCommand::create(data, head() + src_offset * element_size,
                                             size_in_byte(), true);
    }

    [[nodiscard]] BufferDownloadCommand *download_sync(void *data) const noexcept {
        return BufferDownloadCommand::create(data, head(), size_in_byte(), false);
    }

    [[nodiscard]] BufferByteSetCommand *byte_set(uchar value) const noexcept {
        return BufferByteSetCommand::create(head(), size_in_byte(), value, true);
    }

    [[nodiscard]] BufferByteSetCommand *byte_set_sync(uchar value) const noexcept {
        return BufferByteSetCommand::create(head(), size_in_byte(), value, false);
    }

    [[nodiscard]] BufferByteSetCommand *clear() const noexcept {
        return byte_set(0);
    }

    [[nodiscard]] BufferByteSetCommand *clear_sync() const noexcept {
        return byte_set_sync(0);
    }
};

template<typename T = std::byte, int... Dims>
class Buffer : public RHIResource {
    static_assert(is_valid_buffer_element_v<T>);
    static constexpr bool use_for_dsl = is_dsl_basic_v<T>;

public:
    static constexpr size_t element_size = sizeof(T);
    using element_type = T;
    static constexpr std::array<int, sizeof...(Dims)> dims = {Dims...};
    static constexpr bool has_multi_dim() noexcept { return !dims.empty(); }

protected:
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

    template<typename U>
    [[nodiscard]] auto ptr() noexcept {
        if constexpr (std::is_same_v<U, handle_ty>) {
            return handle();
        } else {
            return reinterpret_cast<U>(handle());
        }
    }

    template<typename U = void *>
    [[nodiscard]] auto address(size_t offset) const noexcept {
        return (U)(handle() + offset * element_size);
    }

    void set_size(size_t size) noexcept { _size = size; }

    /// head of the buffer
    [[nodiscard]] handle_ty head() const noexcept { return handle(); }

    /// for dsl trait
    auto operator[](int i) { return T{}; }

    template<typename... Index>
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
        const SubscriptExpr *expr = Function::current()->subscript(Type::of<element_type>(),
                                                                   uniform.expression(),
                                                                   OC_EXPR(index));
        detail::assign(expr, OC_FORWARD(elm));
    }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] detail::AtomicRef<T> atomic(Index &&index) const noexcept {
        const ArgumentBinding &uniform = Function::current()->get_uniform_var(Type::of<decltype(*this)>(),
                                                                              Variable::Tag::BUFFER,
                                                                              memory_block());
        return make_expr<Buffer<T>>(uniform.expression()).atomic(OC_FORWARD(index));
    }

    [[nodiscard]] size_t size() const noexcept { return _size; }
    [[nodiscard]] size_t size_in_byte() const noexcept { return _size * sizeof(T); }

    [[nodiscard]] vector<Command *> reallocate(size_t size, bool async = true) {
        return {BufferReallocateCommand::create(this, size, async),
                HostFunctionCommand::create([this, size] {
                    this->_size = size / element_size;
                }, async)};
    }

    template<typename... Args>
    [[nodiscard]] BufferUploadCommand *upload(Args &&...args) const noexcept {
        return view(0, _size).upload(OC_FORWARD(args)...);
    }

    template<typename... Args>
    [[nodiscard]] BufferDownloadCommand *download(Args &&...args) const noexcept {
        return view(0, _size).download(OC_FORWARD(args)...);
    }

    [[nodiscard]] BufferByteSetCommand *byte_set(uchar value) const noexcept {
        return view(0, _size).byte_set(value);
    }

    [[nodiscard]] BufferByteSetCommand *byte_set_sync(uchar value) const noexcept {
        return view(0, _size).byte_set_sync(value);
    }

    [[nodiscard]] BufferByteSetCommand *clear() const noexcept {
        return byte_set(0);
    }

    [[nodiscard]] BufferByteSetCommand *clear_sync() const noexcept {
        return byte_set_sync(0);
    }

    template<typename Arg>
    requires is_buffer_or_view_v<Arg> && std::is_same_v<buffer_element_t<Arg>, T>
    [[nodiscard]] BufferCopyCommand *copy_from(const Arg &src, uint dst_offset) noexcept {
        return BufferCopyCommand::create(src.head(), head(), 0, dst_offset,
                                         src.size_in_byte(), true);
    }

    template<typename... Args>
    [[nodiscard]] BufferUploadCommand *upload_sync(Args &&...args) const noexcept {
        return view(0, _size).upload_sync(OC_FORWARD(args)...);
    }

    template<typename... Args>
    [[nodiscard]] BufferDownloadCommand *download_sync(Args &&...args) const noexcept {
        return view(0, _size).download_sync(OC_FORWARD(args)...);
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

    void clear_immediately() const noexcept {
        clear_sync()->accept(*_device->command_visitor());
    }
};

template<typename T, int... dims>
BufferView<T, dims...>::BufferView(const Buffer<T, dims...> &buffer)
    : BufferView(buffer.handle(), buffer.size()) {}

}// namespace ocarina