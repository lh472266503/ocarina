//
// Created by Zero on 21/04/2022.
//

#pragma once

#include <utility>

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
        MEMBER,
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
    const Type *type_{};
    uint32_t uid_{};
    Tag tag_;
    string name_{};
    string suffix_{};
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    Variable() noexcept = default;
    Variable(const Type *type, Tag tag, uint uid, string name = "", string suffix = "") noexcept
        : type_(type), tag_(tag), uid_(uid), name_(std::move(name)), suffix_(std::move(suffix)) {}
    [[nodiscard]] constexpr const Type *type() const noexcept { return type_; }
    [[nodiscard]] constexpr Tag tag() const noexcept { return tag_; }
    [[nodiscard]] constexpr uint uid() const noexcept { return uid_; }
    [[nodiscard]] constexpr bool operator==(const Variable &rhs) const noexcept { return uid_ == rhs.uid_; }
    [[nodiscard]] string name() const noexcept;
    void set_name(const char *name) noexcept { name_ = name; }
};

}// namespace ocarina