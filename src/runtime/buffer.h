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

    [[nodiscard]] BufferView<T> subview(size_t offset, size_t size) {
        return BufferView<T>(_handle, _offset + offset, size, _total_size);
    }

    Command *upload(const void *data, size_t size) {

    }

    Command *download(void *data, size_t size) {
    }
};

template<typename T>
class Buffer : public Resource {
private:
    size_t _size{};

public:
    Buffer(Device::Impl *device, size_t size)
        : Resource(device, Tag::BUFFER, device->create_buffer(size)),
          _size(size) {}

    [[nodiscard]] BufferView<T> view(size_t offset, size_t size) {
        return BufferView<T>(_handle, offset, size, _size);
    }
};

}// namespace ocarina