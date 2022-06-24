//
// Created by Zero on 21/06/2022.
//

#pragma once

#include "dsl/var.h"
#include "ast/function.h"

namespace ocarina {

namespace detail {

class IfStmtBuilder {
private:
    IfStmt *_if{nullptr};

public:
    IfStmtBuilder() = default;
    explicit IfStmtBuilder(IfStmt *stmt) : _if(stmt) {}

    template<typename Condition>
    requires concepts::bool_able<expr_value_t<Condition>>
    [[nodiscard]] static IfStmtBuilder create(const Condition &condition) {
        IfStmtBuilder builder(Function::current()->if_(condition.expression()));
        return builder;
    }

    template<typename TrueBranch>
    IfStmtBuilder &operator/(TrueBranch &&true_branch) {
        Function::current()->with(_if->true_branch(), std::forward<TrueBranch>(true_branch));
        return *this;
    }

    template<typename ElseIfCondition>
    requires concepts::bool_able<expr_value_t<ElseIfCondition>>
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
    IfStmtBuilder elif_(ElseIfCondition &&condition, TrueBranch &&true_branch) {
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

namespace detail {

class SwitchStmtBuilder {
private:
    SwitchStmt *_switch_stmt{nullptr};

public:
    explicit SwitchStmtBuilder(SwitchStmt *stmt)
        : _switch_stmt(stmt) {}

    template<typename T>
    requires concepts::switch_able<T>
    [[nodiscard]] static SwitchStmtBuilder create(T &&t) {
        SwitchStmtBuilder builder(Function::current()->switch_(t.expression()));
        return builder;
    }

    template<typename Case>
    [[nodiscard]] SwitchStmtBuilder &operator *(Case && func) {
        Function::current()->with(_switch_stmt->body(), std::forward<Case>(func));
        return *this;
    }

    template<typename Block>
    SwitchStmtBuilder &operator / (Block &&block) {

    }

    template<typename Case>
    SwitchStmtBuilder &case_(Case &&func) {
        return (*this) * std::forward<Case>(func);
    }
};

}// namespace detail

template<typename T>
[[nodiscard]] detail::SwitchStmtBuilder switch_(T &&t) {
    return detail::SwitchStmtBuilder::create(std::forward<T>(t));
}

}// namespace ocarina