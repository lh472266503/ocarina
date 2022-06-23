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

    template<typename ElseIfCondition>
    IfStmtBuilder operator*(ElseIfCondition &&condition) {
        IfStmtBuilder builder;
        Function::current()->with(_if->false_branch(), [&]() {
            builder = create(std::forward<ElseIfCondition>(condition));
        });
        return builder;
    }

    template<typename FalseBranch>
    void operator%(FalseBranch &&false_branch) {
        Function::current()->with(_if->false_branch(), std::forward<FalseBranch>(false_branch));
    }

    template<typename ElseIfCondition, typename TrueBranch>
    IfStmtBuilder elif (ElseIfCondition &&condition, TrueBranch &&true_branch) {
        return (*this) * std::forward<ElseIfCondition>(condition) / std::forward<TrueBranch>(true_branch);
    }

    template<typename FalseBranch>
    void else_(FalseBranch &&false_branch) {
        (*this) % std::forward<FalseBranch>(false_branch);
    }
};
}// namespace detail

template<typename Condition, typename TrueBranch>
detail::IfStmtBuilder if_(Condition &&condition,
                   TrueBranch &&true_branch) {
    return detail::IfStmtBuilder::create(std::forward<Condition>(condition)) / std::forward<TrueBranch>(true_branch);
}

inline void comment(ocarina::string_view str) {
    Function::current()->comment(str);
}

}// namespace ocarina