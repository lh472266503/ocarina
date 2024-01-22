//
// Created by Zero on 22/01/2024.
//

#pragma once

#include "core/stl.h"
#include "type.h"
#include "core/concepts.h"
#include "usage.h"
#include "variable.h"
#include "op.h"

namespace ocarina {

class Function;

class ASTNode {
protected:
    Function *_context{};

public:
    OC_MAKE_MEMBER_GETTER_SETTER(context, )
    virtual bool check_context(const Function *ctx) const noexcept {
        OC_ASSERT(_context == ctx);
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
