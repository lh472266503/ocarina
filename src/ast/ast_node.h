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

#define OC_MAKE_CHECK_CONTEXT_ELEMENT(name) (name).check_context(ctx)

#define OC_MAKE_CHECK_CONTEXT(...)                                      \
    bool check_context(Function *ctx) const noexcept {                  \
        return true && MAP(OC_MAKE_CHECK_CONTEXT_ELEMENT, __VA_ARGS__); \
    }

class ASTNode {
protected:
    Function *_context{};

public:
    OC_MAKE_MEMBER_GETTER_SETTER(context, )
    virtual bool check_context(Function *ctx) const noexcept {
        return true;
    }
};

}// namespace ocarina
