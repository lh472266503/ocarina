//
// Created by Zero on 21/04/2022.
//

#pragma once

#include "core/stl.h"
#include "core/hash.h"

namespace ocarina {

class Variable {
public:
    enum struct Tag : uint32_t {
        // data
        LOCAL,
        SHARED,

        // reference
        REFERENCE,

        // resources
        BUFFER,
        TEXTURE,
        BINDLESS_ARRAY,
        ACCEL,

        // builtins
        THREAD_ID,
        BLOCK_ID,
        DISPATCH_ID,
        DISPATCH_SIZE
    };

private:
    const Type *_type;
    uint32_t _uid;
    Tag _tag;

public:
    Variable() noexcept = default;
    Variable(const Type *type, Tag tag, uint uid) noexcept
        : _type(type), _tag(tag), _uid(uid) {}
    [[nodiscard]] const Type *type() const noexcept { return _type; }
    [[nodiscard]] Tag tag() const noexcept { return _tag; }
    [[nodiscard]] uint uid() const noexcept { return _uid; }
    [[nodiscard]] bool operator==(Variable rhs) const noexcept { return _uid == rhs._uid; }
    [[nodiscard]] uint64_t hash() const noexcept {
        auto u0 = static_cast<uint64_t>(_uid);
        auto u1 = static_cast<uint64_t>(_tag);
        using namespace std::string_view_literals;
        return hash64(u0 | (u1 << 32u), hash64(_type->hash(), hash64("__hash_variable"sv)));
    }
};

}// namespace ocarina