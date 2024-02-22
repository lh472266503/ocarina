//
// Created by Zero on 21/04/2022.
//

#pragma once

#include "core/stl.h"
#include "core/hash.h"
#include "type.h"

namespace ocarina {

class OC_AST_API Variable : public Hashable {
public:
    enum struct Tag : uint32_t {
        // data
        LOCAL,
        SHARED,
        UNIFORM,

        // reference
        REFERENCE,

        // resources
        BUFFER,
        BYTE_BUFFER,
        TEXTURE,
        BINDLESS_ARRAY,
        ACCEL,

        // builtins
        THREAD_IDX,
        BLOCK_IDX,
        THREAD_ID,
        DISPATCH_IDX,
        DISPATCH_ID,
        DISPATCH_DIM
    };

private:
    const Type *_type{};
    uint32_t _uid{};
    Tag _tag;
    const char *_name{};
    const char *_suffix{};
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    Variable() noexcept = default;
    Variable(const Type *type, Tag tag, uint uid, const char *name = nullptr, const char *suffix = nullptr) noexcept
        : _type(type), _tag(tag), _uid(uid), _name(name),_suffix(suffix) {}
    [[nodiscard]] constexpr const Type *type() const noexcept { return _type; }
    [[nodiscard]] constexpr Tag tag() const noexcept { return _tag; }
    [[nodiscard]] constexpr uint uid() const noexcept { return _uid; }
    [[nodiscard]] constexpr bool operator==(const Variable &rhs) const noexcept { return _uid == rhs._uid; }
    [[nodiscard]] string name() const noexcept;
    void set_name(const char *name) noexcept { _name = name; }
};

}// namespace ocarina