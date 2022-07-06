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
    Buffer(Device::Impl *device, handle_ty handle, size_t size)
        : Resource(device, Tag::BUFFER, handle),
          _size_in_bytes(size) {}
};

}// namespace ocarina