//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "resource.h"

namespace ocarina {

class RawBuffer : public Resource {
private:
    size_t _size_in_bytes{};

public:
    RawBuffer(handle_ty handle, size_t size)
        : Resource(Tag::BUFFER, handle),
          _size_in_bytes(size) {}
    [[nodiscard]] auto size_in_bytes() const noexcept { return _size_in_bytes; }
    void download(void *host_ptr, size_t size, size_t offset) noexcept;
    void upload(const void *host_ptr, size_t size, size_t offset) noexcept;
};

template<typename T>
class Buffer : public RawBuffer {
};

}// namespace ocarina