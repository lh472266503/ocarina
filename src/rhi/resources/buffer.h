//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "resource.h"
#include "rhi/command.h"
#include "rhi/stats.h"
#include "bindless_array.h"

namespace ocarina {

template<typename T, int... Dims>
class Buffer;

template<typename T, int... Dims>
class BufferView {
private:
    size_t _element_size{Buffer<T>::calculate_size()};
    handle_ty _handle{};
    size_t _offset{};
    size_t _size{};
    size_t _total_size{};

public:
    BufferView() = default;
    BufferView(const Buffer<T, Dims...> &buffer);
    [[nodiscard]] handle_ty handle() const { return _handle; }
    [[nodiscard]] size_t size() const { return _size; }
    [[nodiscard]] size_t element_size() const noexcept { return _element_size; }
    [[nodiscard]] size_t size_in_byte() const noexcept { return _size * _element_size; }
    BufferView(handle_ty handle, size_t offset, size_t size, size_t total_size)
        : _handle(handle), _offset(offset), _size(size), _total_size(total_size) {}

    BufferView(handle_ty handle, size_t total_size)
        : _handle(handle), _offset(0), _total_size(total_size), _size(total_size) {}

    [[nodiscard]] handle_ty head() const { return _handle + _offset * _element_size; }

    [[nodiscard]] BufferView<T> subview(size_t offset, size_t size) const noexcept {
        return BufferView<T>(_handle, _offset + offset, size, _total_size);
    }

    template<typename Arg>
    requires is_buffer_or_view_v<Arg> && std::is_same_v<buffer_element_t<Arg>, T>
    [[nodiscard]] BufferCopyCommand *copy_from(const Arg &src, uint dst_offset = 0) noexcept {
        return BufferCopyCommand::create(src.head(), head(), 0, dst_offset * _element_size,
                                         src.size_in_byte(), true);
    }

    template<typename Arg>
    requires is_buffer_or_view_v<Arg> && std::is_same_v<buffer_element_t<Arg>, T>
    [[nodiscard]] BufferCopyCommand *copy_to(Arg &dst, uint src_offset = 0) noexcept {
        return BufferCopyCommand::create(head(), dst.head(), src_offset * _element_size,
                                         0, dst.size_in_byte(), true);
    }

    [[nodiscard]] BufferUploadCommand *upload(const void *data, bool async = true) const noexcept {
        return BufferUploadCommand::create(data, head(), _size * _element_size, async);
    }

    [[nodiscard]] BufferUploadCommand *upload_sync(const void *data) const noexcept {
        return upload(data, false);
    }

    [[nodiscard]] BufferDownloadCommand *download(void *data, uint src_offset = 0, bool async = true) const noexcept {
        return BufferDownloadCommand::create(data, head() + src_offset * _element_size,
                                             size_in_byte(), async);
    }

    [[nodiscard]] BufferDownloadCommand *download_sync(void *data) const noexcept {
        return download(data, 0, false);
    }

    [[nodiscard]] BufferByteSetCommand *byte_set(uchar value, bool async = true) const noexcept {
        return BufferByteSetCommand::create(head(), size_in_byte(), value, async);
    }

    [[nodiscard]] BufferByteSetCommand *byte_set_sync(uchar value) const noexcept {
        return byte_set(value, false);
    }

    [[nodiscard]] BufferByteSetCommand *reset(bool async = true) const noexcept {
        return byte_set(0, async);
    }

    [[nodiscard]] BufferByteSetCommand *reset_sync() const noexcept {
        return byte_set_sync(0);
    }
};

template<typename T = std::byte, int... Dims>
class Buffer : public RHIResource {
    static_assert(is_valid_buffer_element_v<T>);
    static constexpr bool use_for_dsl = is_dsl_basic_v<T>;

public:
    using element_type = T;
    static constexpr ocarina::array<int, sizeof...(Dims)> dims = {Dims...};
    static constexpr bool has_multi_dim() noexcept { return !dims.empty(); }

protected:
    size_t _size{};
    size_t _element_size{0};

    /// just for construct memory block
    mutable BufferProxy<T> _proxy{};

public:
    Buffer() : _element_size(calculate_size()) {}

    [[nodiscard]] size_t element_size() const noexcept {
        return _element_size;
    }

    Buffer(Device::Impl *device, size_t size, const string &desc = "")
        : RHIResource(device, Tag::BUFFER, device->create_buffer(size * calculate_size(), desc)),
          _size(size), _element_size(calculate_size()) {}

    static size_t calculate_size() noexcept {
        if constexpr (is_struct_v<T>) {
            return Type::of<T>()->size();
        }
        return sizeof(T);
    }

    Buffer(BufferView<T, Dims...> buffer_view)
        : RHIResource(nullptr, Tag::BUFFER, buffer_view.head()),
          _size(buffer_view.size()), _element_size(calculate_size()) {}

    void destroy() override {
        _destroy();
        _size = 0;
    }

    [[nodiscard]] BufferView<T> view(size_t offset = 0, size_t size = 0) const noexcept {
        size = size == 0 ? _size - offset : size;
        return BufferView<T>(_handle, offset, size, _size);
    }

    // Move constructor
    Buffer(Buffer &&other) noexcept
        : RHIResource(std::move(other)) {
        this->_size = other._size;
        this->_element_size = other._element_size;
    }

    // Move assignment
    Buffer &operator=(Buffer &&other) noexcept {
        destroy();
        RHIResource::operator=(std::move(other));
        this->_size = other._size;
        this->_element_size = other._element_size;
        return *this;
    }

    [[nodiscard]] const void *proxy_ptr() const noexcept {
        _proxy.ptr = reinterpret_cast<T *>(_handle);
        _proxy.size = static_cast<uint>(_size);
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
        return (U)(handle() + offset * _element_size);
    }

    void set_size(size_t size) noexcept { _size = size; }

    /// head of the buffer
    [[nodiscard]] handle_ty head() const noexcept { return handle(); }

    /// for dsl trait
    auto operator[](int i) { return T{}; }

    template<typename U = T>
    [[nodiscard]] const Expression *expression() const noexcept {
        const CapturedResource &captured_resource = Function::current()->get_captured_resource(Type::of<decltype(*this)>(),
                                                                                               Variable::Tag::BUFFER,
                                                                                               memory_block());
        return captured_resource.expression();
    }

    template<typename Index>
    requires concepts::all_integral<expr_value_t<Index>>
    OC_NODISCARD auto
    read(Index &&index, bool check_boundary = true) const {
        auto expr = make_expr<Buffer<T>>(expression());
        if (check_boundary) {
            return expr.read_and_check(OC_FORWARD(index),
                                       static_cast<uint>(_size),
                                       typeid(*this).name());
        } else {
            return expr.read(OC_FORWARD(index));
        }
    }

    template<typename... Index>
    requires concepts::all_integral<expr_value_t<Index>...>
    OC_NODISCARD auto
    read_multi(Index &&...index) const {
        return make_expr<Buffer<T>>(expression()).read(OC_FORWARD(index)...);
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<element_type, expr_value_t<Val>>
    void write(Index &&index, Val &&elm, bool check_boundary = true) {
        auto expr = make_expr<Buffer<T>>(expression());
        if (check_boundary) {
            expr.write_and_check(OC_FORWARD(index), OC_FORWARD(elm),
                                 static_cast<uint>(_size), typeid(*this).name());
        } else {
            expr.write(OC_FORWARD(index), OC_FORWARD(elm));
        }
    }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] detail::AtomicRef<T> atomic(Index &&index) const noexcept {
        return make_expr<Buffer<T>>(expression()).atomic(OC_FORWARD(index));
    }

    [[nodiscard]] size_t size() const noexcept { return _size; }
    [[nodiscard]] size_t size_in_byte() const noexcept { return _size * sizeof(T); }

    [[nodiscard]] CommandList reallocate(size_t size, bool async = true) {
        return {BufferReallocateCommand::create(this, size * _element_size, async),
                HostFunctionCommand::create([this, size] {
                    this->_size = size;
                },
                                            async)};
    }

    void reallocate_immediately(size_t size) {
        CommandList command_list = reallocate(size, false);
        command_list.accept(*_device->command_visitor());
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

    [[nodiscard]] BufferByteSetCommand *reset() const noexcept {
        return byte_set(0);
    }

    [[nodiscard]] BufferByteSetCommand *reset_sync() const noexcept {
        return byte_set_sync(0);
    }

    template<typename... Args>
    [[nodiscard]] BufferCopyCommand *copy_from(Args &&...args) const noexcept {
        return view(0, _size).copy_from(OC_FORWARD(args)...);
    }

    template<typename... Args>
    [[nodiscard]] BufferCopyCommand *copy_to(Args &&...args) const noexcept {
        return view(0, _size).copy_to(OC_FORWARD(args)...);
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

    void reset_immediately() const noexcept {
        reset_sync()->accept(*_device->command_visitor());
    }
};

template<typename T, int... dims>
BufferView<T, dims...>::BufferView(const Buffer<T, dims...> &buffer)
    : BufferView(buffer.handle(), buffer.size()) {}

class ByteBuffer : public RHIResource {
private:
    /// just for construct memory block
    mutable BufferProxy<uchar> _proxy{};
    size_t _size{};

public:
    ByteBuffer(Device::Impl *device, size_t size, const string &desc = "")
        : RHIResource(device, Tag::BUFFER, device->create_buffer(size, desc)),
          _size(size) {}
};

}// namespace ocarina