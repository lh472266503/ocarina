//
// Created by Zero on 03/07/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"

namespace ocarina {

using handle_ty = uint64_t;

class Resource : public concepts::Noncopyable {
public:
    enum Tag : uint8_t {
        BUFFER,
        TEXTURE
    };

private:
    Tag _tag;
    handle_ty _handle{};

public:
    Resource(Tag tag, handle_ty handle) : _tag(tag), _handle(handle) {}
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
};
}// namespace ocarina