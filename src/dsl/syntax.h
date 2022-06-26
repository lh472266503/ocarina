//
// Created by Zero on 21/06/2022.
//

#pragma once

#include "dsl/var.h"
#include "dsl/expr.h"
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

    template<typename CaseExpr, typename Body>
    SwitchStmtBuilder &case_(CaseExpr &&case_expr, Body &&body) noexcept {
        Function::current()->with(_switch_stmt->body(), [&] {
            CaseStmtBuilder::create(std::forward<CaseExpr>(case_expr)) * std::forward<Body>(body);
        });
        return *this;
    }

    template<typename Body>
    void default_(Body &&body) noexcept {
        Function::current()->with(_switch_stmt->body(), [&] {
            DefaultStmtBuilder() * std::forward<Body>(body);
        });
    }

    template<typename Body>
    SwitchStmtBuilder &operator*(Body &&func) &&noexcept {
        Function::current()->with(_switch_stmt->body(), std::forward<Body>(func));
        return *this;
    }
};

}// namespace detail

template<typename T, typename Body>
void switch_(T &&t, Body &&body) noexcept {
    detail::SwitchStmtBuilder::create(std::forward<T>(t)) * std::forward<Body>(body);
}

template<typename T>
decltype(auto) switch_(T &&t) noexcept {
    return detail::SwitchStmtBuilder::create(std::forward<T>(t));
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

namespace detail {
class LoopStmtBuilder {
private:
    LoopStmt *_loop{};

public:
    explicit LoopStmtBuilder(LoopStmt *loop) noexcept : _loop(loop) {}

    static auto create() noexcept {
        return LoopStmtBuilder(Function::current()->loop());
    }

    template<typename Func>
    LoopStmtBuilder &operator/(Func &&func) noexcept {
        Function::current()->with(_loop->body(), std::forward<Func>(func));
        return *this;
    }

    template<typename Body>
    void operator*(Body &&body) noexcept {
        Function::current()->with(_loop->body(), std::forward<Body>(body));
    }
};
}// namespace detail

template<typename Body>
void loop(Body &&body) noexcept {
    detail::LoopStmtBuilder::create() * std::forward<Body>(body);
}

template<typename Condition, typename Body>
void while_(Condition &&cond, Body &&body) noexcept {
    detail::LoopStmtBuilder::create() * [&]() noexcept {
        if constexpr (std::is_invocable_v<Condition>) {
            if_(!cond(), [&] {
                break_();
            });
        } else {
            if_(!cond, [&] {
                break_();
            });
        }
        body();
    };
}

namespace detail {
template<typename T = int>
class ForStmtBuilder {
private:
    Var<T> _var;
    Var<T> _begin;
    Var<T> _end;
    Var<T> _step;
    ForStmt *_for_stmt;

public:
    explicit ForStmtBuilder(ForStmt *for_stmt) noexcept : _for_stmt(for_stmt) {}

    ForStmtBuilder(const Var<T> &begin, const Var<T> &end, const Var<T> &step)
        : _begin(begin),
          _end(end),
          _step(step) {
        _var = _begin;
        _for_stmt = Function::current()->for_(_var.expression(),
                                              extract_expression(_var < _end),
                                              _step.expression());
    }

    static ForStmtBuilder create(Var<T> var, const Expr<bool> &Condition, Var<T> step) noexcept {
        return ForStmtBuilder(Function::current()->for_(
            extract_expression(var),
            extract_expression(Condition),
            extract_expression(step)));
    }

    template<typename Body>
    void operator/(Body &&body) noexcept {
        Function::current()->with(_for_stmt->body(), [&]() noexcept {
            body(_var);
        });
    }
};
}// namespace detail

template<typename Count>
requires concepts::integral<expr_value_t<Count>>
auto range(Count &&count) noexcept {
    return detail::ForStmtBuilder<expr_value_t<Count>>(0, def_expr(std::forward<Count>(count)), 1);
}

template<typename Begin, typename End>
requires concepts::integral<expr_value_t<Begin>>
auto range(Begin &&begin, End &&end) noexcept {
    return detail::ForStmtBuilder<expr_value_t<Begin>>(def_expr(std::forward<Begin>(begin)),
                                                       def_expr(std::forward<End>(end)),
                                                       1);
}

template<typename Begin, typename End, typename Step>
requires concepts::integral<expr_value_t<Begin>>
auto range(Begin &&begin, End &&end, Step &&step) noexcept {
    return detail::ForStmtBuilder<expr_value_t<Begin>>(def_expr(std::forward<Begin>(begin)),
                                                       def_expr(std::forward<End>(end)),
                                                       def_expr(std::forward<Step>(step)));
}

template<typename Count, typename Body>
requires concepts::integral<expr_value_t<Count>>
void for_range(Count &&count, Body &&body) noexcept {
    range(std::forward<Count>(count)) / std::forward<Body>(body);
}

//template<typename ...Args>

}// namespace ocarina