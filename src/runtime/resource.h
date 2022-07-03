//
// Created by Zero on 03/07/2022.
//

#pragma once

#include "core/stl.h"

namespace ocarina {

class Resource {
public:
    enum Tag : uint8_t {
        BUFFER,
        TEXTURE
    };

    using handle_type = uint64_t;

private:
    Tag _tag;
    handle_type _handle{};

public:
    Resource(Tag tag, handle_type handle) : _tag(tag), _handle(handle) {}
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
};
}// namespace ocarina