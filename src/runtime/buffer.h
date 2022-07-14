//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "resource.h"
#include "command.h"

namespace ocarina {

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
    BufferView(handle_ty handle, size_t offset, size_t size, size_t total_size)
        : _handle(handle), _offset(offset), _size(size), _total_size(total_size) {}

    BufferView(handle_ty handle, size_t total_size)
        : _handle(handle), _offset(0), _total_size(total_size), _size(total_size) {}

    [[nodiscard]] BufferView<T> subview(size_t offset, size_t size) const noexcept {
        return BufferView<T>(_handle, _offset + offset, size, _total_size);
    }

    [[nodiscard]] BufferUploadCommand *upload(const void *data) const noexcept {
        return BufferUploadCommand::create(data, _handle + _offset * element_size,
                                           _size * element_size);
    }

    [[nodiscard]] BufferDownloadCommand *download(void *data) const noexcept {
        return BufferDownloadCommand::create(data, _handle + _offset * element_size,
                                             _size * element_size);
    }
};

template<typename T>
class Buffer : public Resource {
public:
    static constexpr size_t element_size = sizeof(T);
private:
    size_t _size{};

public:
    Buffer(Device::Impl *device, size_t size)
        : Resource(device, Tag::BUFFER, device->create_buffer(size * sizeof(T))),
          _size(size) {}

    [[nodiscard]] BufferView<T> view(size_t offset, size_t size) const noexcept {
        return BufferView<T>(_handle, offset, size, _size);
    }

    template<typename... Args>
    [[nodiscard]] BufferUploadCommand *upload(Args &&...args) const noexcept {
        return view(0, _size).upload(OC_FORWARD(args)...);
    }

    template<typename... Args>
    [[nodiscard]] BufferDownloadCommand *download(Args &&...args) const noexcept {
        return view(0, _size).download(OC_FORWARD(args)...);
    }
};

namespace detail {

template<typename T>
struct is_buffer_impl : std::false_type {};

template<typename T>
struct is_buffer_impl<Buffer<T>> : std::true_type {};

template<typename T>
struct is_buffer_view_impl : std::false_type {};

template<typename T>
struct is_buffer_view_impl<BufferView<T>> : std::true_type {};

template<typename T>
struct buffer_element_impl {
    using type = T;
};

template<typename T>
struct buffer_element_impl<Buffer<T>> {
    using type = T;
};

template<typename T>
struct buffer_element_impl<BufferView<T>> {
    using type = T;
};

}// namespace detail

template<typename T>
using is_buffer = detail::is_buffer_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_buffer_view = detail::is_buffer_view_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_buffer_or_view = std::disjunction<is_buffer<T>, is_buffer_view<T>>;

template<typename T>
constexpr auto is_buffer_v = is_buffer<T>::value;

template<typename T>
constexpr auto is_buffer_view_v = is_buffer_view<T>::value;

template<typename T>
constexpr auto is_buffer_or_view_v = is_buffer_or_view<T>::value;

}// namespace ocarina