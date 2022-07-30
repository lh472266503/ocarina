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
                                           _size * element_size, true);
    }

    [[nodiscard]] BufferUploadCommand *upload_sync(const void *data) const noexcept {
        return BufferUploadCommand::create(data, _handle + _offset * element_size,
                                           _size * element_size, false);
    }

    [[nodiscard]] BufferDownloadCommand *download(void *data) const noexcept {
        return BufferDownloadCommand::create(data, _handle + _offset * element_size,
                                             _size * element_size, true);
    }

    [[nodiscard]] BufferDownloadCommand *download_sync(void *data) const noexcept {
        return BufferDownloadCommand::create(data, _handle + _offset * element_size,
                                             _size * element_size, false);
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

    /// for dsl trait
    auto operator[](int i) { return T{}; }

    template<typename Index>
    requires ocarina::is_integral_v<expr_value_t<Index>>
    [[nodiscard]] auto read(Index &&index) {
        Function::current()->add_uniform_var(Type::of<Buffer<T>>(), handle());
        return ;
    }

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

}// namespace ocarina