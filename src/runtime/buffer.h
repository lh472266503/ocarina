//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "resource.h"
#include "command.h"

namespace ocarina {

template<typename T>
class BufferView {
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

    [[nodiscard]] BufferUploadCommand *upload(const void *data, size_t offset = 0, size_t size = 0) const noexcept {
        size = size == 0 ? _size : size;
        offset *= sizeof(T);
        return BufferUploadCommand::create(data, offset,
                                           _handle + _offset * sizeof(T),
                                           size * sizeof(T));
    }

    [[nodiscard]] BufferDownloadCommand *download(void *data, size_t offset = 0, size_t size = 0) const noexcept {
        size = size == 0 ? _size : size;
        offset *= sizeof(T);
        return BufferDownloadCommand::create(data, offset, _handle, size * sizeof(T));
    }
};

template<typename T>
class Buffer : public Resource {
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

}// namespace ocarina