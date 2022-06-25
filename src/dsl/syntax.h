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
    [[nodiscard]] static IfStmtBuilder create(Condition &&condition) noexcept {
        IfStmtBuilder builder(Function::current()->if_(extract_expression(std::forward<Condition>(condition))));
        return builder;
    }

    template<typename TrueBranch>
    IfStmtBuilder &operator/(TrueBranch &&true_branch) noexcept {
        Function::current()->with(_if->true_branch(), std::forward<TrueBranch>(true_branch));
        return *this;
    }

    template<typename ElseIfCondition>
    requires concepts::bool_able<expr_value_t<ElseIfCondition>>
        IfStmtBuilder operator*(ElseIfCondition &&condition) noexcept {
        IfStmtBuilder builder;
        Function::current()->with(_if->false_branch(), [&]() {
            builder = create(std::forward<ElseIfCondition>(condition));
        });
        return builder;
    }

    template<typename FalseBranch>
    void operator%(FalseBranch &&false_branch) noexcept {
        Function::current()->with(_if->false_branch(), std::forward<FalseBranch>(false_branch));
    }

    template<typename ElseIfCondition, typename TrueBranch>
    IfStmtBuilder elif_(ElseIfCondition &&condition, TrueBranch &&true_branch) noexcept {
        return (*this) * std::forward<ElseIfCondition>(condition) / std::forward<TrueBranch>(true_branch);
    }

    template<typename FalseBranch>
    void else_(FalseBranch &&false_branch) noexcept {
        (*this) % std::forward<FalseBranch>(false_branch);
    }
};
}// namespace detail

template<typename Condition, typename TrueBranch>
detail::IfStmtBuilder if_(Condition &&condition,
                          TrueBranch &&true_branch) noexcept {
    return detail::IfStmtBuilder::create(std::forward<Condition>(condition)) / std::forward<TrueBranch>(true_branch);
}

inline void comment(ocarina::string_view str) noexcept {
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
    requires concepts::switch_able<expr_value_t<T>>
    [[nodiscard]] static SwitchStmtBuilder create(T &&t) noexcept {
        SwitchStmtBuilder builder(Function::current()->switch_(t.expression()));
        return builder;
    }

    template<typename Body>
    SwitchStmtBuilder &operator*(Body &&func) noexcept {
        Function::current()->with(_switch_stmt->body(), std::forward<Body>(func));
        return *this;
    }
};

class CaseStmtBuilder {
private:
    SwitchCaseStmt *_case_stmt{nullptr};

public:
    explicit CaseStmtBuilder(SwitchCaseStmt *stmt)
        : _case_stmt(stmt) {}

    template<typename CaseExpr>
    [[nodiscard]] static CaseStmtBuilder create(CaseExpr &&case_expr) noexcept {
        CaseStmtBuilder builder(Function::current()->switch_case(extract_expression(std::forward<CaseExpr>(case_expr))));
        return builder;
    }

    template<typename Body>
    void operator*(Body &&body) noexcept {
        Function::current()->with(_case_stmt->body(), std::forward<Body>(body));
    }
};

class DefaultStmtBuilder {
private:
    SwitchDefaultStmt *_default_stmt{};

public:
    DefaultStmtBuilder() noexcept
        : _default_stmt(Function::current()->switch_default()) {}

    template<typename Body>
    void operator*(Body &&body) noexcept {
        Function::current()->with(_default_stmt->body(), std::forward<Body>(body));
    }
};

}// namespace detail

template<typename T, typename Body>
void switch_(T &&t, Body &&body) noexcept {
    detail::SwitchStmtBuilder::create(std::forward<T>(t)) * std::forward<Body>(body);
}

template<typename T, typename Body>
void case_(T &&t, Body &&body) noexcept {
    detail::CaseStmtBuilder::create(std::forward<T>(t)) * std::forward<Body>(body);
}

inline void break_() noexcept {
    Function::current()->break_();
}

template<typename Body>
void default_(Body &&body) noexcept {
    detail::DefaultStmtBuilder() * std::forward<Body>(body);
}

inline void continue_() noexcept {
    Function::current()->continue_();
}

}// namespace ocarina