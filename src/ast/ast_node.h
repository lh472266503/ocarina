//
// Created by Zero on 22/01/2024.
//

#pragma once

#include "core/stl.h"
#include "core/logging.h"
#include "type.h"
#include "core/concepts.h"
#include "variable.h"
#include "op.h"

namespace ocarina {

enum struct Usage : uint32_t {
    NONE = 0u,
    READ = 1 << 0,
    WRITE = 1 << 1,
    READ_WRITE = READ | WRITE
};

//OC_MAKE_ENUM_BIT_OPS(Usage, |, &, <<, >>)

//inline auto operator|(Usage lhs, Usage rhs) { return static_cast<Usage>(ocarina::to_underlying(lhs) | ocarina::to_underlying(rhs)); }
//inline auto operator&(Usage lhs, Usage rhs) { return static_cast<Usage>(ocarina::to_underlying(lhs) & ocarina::to_underlying(rhs)); }
//inline auto operator<<(Usage lhs, Usage rhs) { return static_cast<Usage>(ocarina::to_underlying(lhs) << ocarina::to_underlying(rhs)); }
//inline auto operator>>(Usage lhs, Usage rhs) { return static_cast<Usage>(ocarina::to_underlying(lhs) >> ocarina::to_underlying(rhs)); }


//[[nodiscard]] inline bool is_write(Usage usage) {
//    return (usage & Usage::WRITE) == Usage::WRITE;
//}

class Function;

class ASTNode {
protected:
    Function *_context{};

public:
    OC_MAKE_MEMBER_GETTER_SETTER(context, )
    virtual bool check_context(const Function *ctx) const noexcept {
        if (ctx != _context) {
            volatile int i = 0;
        }
        return _context == ctx;
    }
};

namespace detail {
template<typename T>
bool check_context(const T &t, const Function *ctx) {
    if constexpr (concepts::iterable<T>) {
        bool ret = true;
        for (const auto &item : t) {
            ret = ret && check_context(item, ctx);
        }
        return ret;
    } else if constexpr (requires() {
                             t.check_context(ctx);
                         }) {
        return t.check_context(ctx);
    } else if constexpr (requires() {
                             t->check_context(ctx);
                         }) {
        if (t == nullptr) {
            return true;
        }
        return t->check_context(ctx);
    } else {
        static_assert(always_false_v<T>);
    }
}
}// namespace detail

#define OC_MAKE_CHECK_CONTEXT_ELEMENT(name) &&detail::check_context((name), ctx)

#define OC_MAKE_CHECK_CONTEXT(Super, ...)                                                 \
    bool check_context(const Function *ctx) const noexcept override {                     \
        return Super::check_context(ctx) MAP(OC_MAKE_CHECK_CONTEXT_ELEMENT, __VA_ARGS__); \
    }

}// namespace ocarina
