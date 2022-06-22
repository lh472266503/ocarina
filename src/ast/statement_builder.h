//
// Created by Zero on 21/06/2022.
//

#pragma once

#include "dsl/var.h"
#include "function.h"
#include "expression.h"

namespace ocarina {
class IfStmt;

class IfStmtBuilder {
private:
    IfStmt *_if;

public:
    explicit IfStmtBuilder(IfStmt *stmt) : _if(stmt) {}

    static IfStmtBuilder create(const Computable<bool> &condition) {
        IfStmtBuilder builder(Function::current()->if_(condition.expression()));
        return builder;
    }

    template<typename TrueBranch>
    IfStmtBuilder &operator/(TrueBranch &&true_branch) {
        Function::current()->with(_if->true_branch(), std::forward<TrueBranch>(true_branch));
        return *this;
    }

    template<typename FalseBranch>
    IfStmtBuilder &operator%(FalseBranch &&false_branch) {
        Function::current()->with(_if->false_branch(), std::forward<FalseBranch>(false_branch));
        return *this;
    }
};

}// namespace ocarina