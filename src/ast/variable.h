//
// Created by Zero on 21/04/2022.
//

#pragma once

#include <utility>
#include "core/stl.h"
#include "core/hash.h"
#include "type.h"

namespace ocarina {

enum struct Usage : uint32_t {
    NONE = 0u,
    READ = 1 << 0,
    WRITE = 1 << 1,
    READ_WRITE = READ | WRITE
};

[[nodiscard]] inline bool is_write(Usage usage) {
    return (to_underlying(usage) & to_underlying(Usage::WRITE)) == to_underlying(Usage::WRITE);
}

class Function;

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
    const Function *_context{nullptr};
    uint32_t uid_{};
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;
    friend class Function;
    Variable(const Function *context,
             const Type *type, Tag tag, uint uid,
             string name = "", string suffix = "") noexcept;

private:
    [[nodiscard]] string get_name() const noexcept;
    [[nodiscard]] string get_suffix() const noexcept;

public:
    Variable() noexcept = default;
    [[nodiscard]] constexpr const Type *type() const noexcept { return type_; }
    [[nodiscard]] Tag tag() const noexcept;
    [[nodiscard]] constexpr uint uid() const noexcept { return uid_; }

    [[nodiscard]] constexpr bool operator==(const Variable &rhs) const noexcept { return uid_ == rhs.uid_; }
    [[nodiscard]] string name() const noexcept;
    [[nodiscard]] Usage usage() const noexcept;
    void set_tag(Tag tag) noexcept;
    void set_name(string name) noexcept;
    void set_suffix(string suffix) noexcept;
};

struct VariableData {
    Usage usage;
    Variable::Tag tag{};
    string name{};
    string suffix{};
    explicit VariableData(Usage u,
                          Variable::Tag t = Variable::Tag::LOCAL)
        : usage(u), tag(t) {}
};

}// namespace ocarina