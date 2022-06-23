//
// Created by Zero on 21/06/2022.
//

#pragma once

#include "dsl/var.h"
#include "ast/function.h"
#include "ast/expression.h"

namespace ocarina {
class IfStmt;

namespace detail {

class IfStmtBuilder {
private:
    IfStmt *_if{nullptr};

public:
    IfStmtBuilder() = default;
    explicit IfStmtBuilder(IfStmt *stmt) : _if(stmt) {}

    template<typename Condition>
    static IfStmtBuilder create(const Condition &condition) {
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

    template<typename ElseIfCondition>
    auto operator*(const ElseIfCondition &condition) {
        IfStmtBuilder builder;
        Function::current()->with(_if->false_branch(), [&]() {
            builder = create(condition);
        });
        return builder;
    }
};
}// namespace detail

template<typename Condition>
[[nodiscard]] decltype(auto) if_(const Condition &condition) {
    return detail::IfStmtBuilder::create(condition);
}

inline void comment(ocarina::string_view str) {
    Function::current()->comment(str);
}

}// namespace ocarina