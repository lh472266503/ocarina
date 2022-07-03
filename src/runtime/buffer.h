//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "resource.h"

namespace ocarina {
template<typename T>
class Buffer : public Resource {
private:
    size_t _size_in_bytes{};

public:
    Buffer(handle_ty handle, size_t size)
        : Resource(Tag::BUFFER, handle),
          _size_in_bytes(size) {}
    [[nodiscard]] auto size_in_bytes() const noexcept { return _size_in_bytes; }

};
}// namespace ocarina