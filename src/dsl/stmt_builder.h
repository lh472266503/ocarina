//
// Created by Zero on 21/06/2022.
//

#pragma once

#include "dsl/var.h"
#include "dsl/expr.h"
#include "ast/function.h"
#include "operators.h"

namespace ocarina {

inline void comment(const ocarina::string &str) noexcept {
    Function::current()->comment(str);
}

namespace detail {
template<typename Lhs, typename Rhs>
requires (!is_param_struct_v<expr_value_t<Lhs>> && !is_param_struct_v<expr_value_t<Rhs>>)
inline void assign(Lhs &&lhs, Rhs &&rhs) noexcept {
    if constexpr (concepts::assign_able<expr_value_t<Lhs>, expr_value_t<Rhs>>) {
        Function::current()->assign(
            detail::extract_expression(std::forward<Lhs>(lhs)),
            detail::extract_expression(std::forward<Rhs>(rhs)));
    } else if constexpr (std::is_pointer_v<std::remove_cvref_t<Rhs>>) {
        Function::current()->assign(OC_EXPR(lhs), rhs);
    } else if constexpr (std::is_pointer_v<std::remove_cvref_t<Lhs>>) {
        Function::current()->assign(lhs, OC_EXPR(rhs));
    } else {
        static_assert(always_false_v<Lhs, Rhs>, "invalid assignment operator !");
    }
}
}// namespace detail

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> eval(T &&x) noexcept {
    return Var(OC_FORWARD(x));
}

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> eval(const Expression *expr) noexcept {
    using RawType = expr_value_t<T>;
    return Var<RawType>(Expr<RawType>(expr));
}

template<typename T>
[[nodiscard]] inline Expr<expr_value_t<T>> make_expr(T &&x) noexcept {
    if constexpr (is_expr_v<T>) {
        return make_expr<T>(x.expression());
    } else {
        return Expr<expr_value_t<T>>(std::forward<T>(x));
    }
}

template<typename T>
[[nodiscard]] inline Expr<expr_value_t<T>> make_expr(const Expression *expr) noexcept {
    using RawType = expr_value_t<T>;
    return Expr<RawType>(expr);
}

template<typename T, typename Arg>
[[nodiscard]] inline auto cast(Arg &&arg) {
    if constexpr (is_dsl_v<Arg>) {
        return arg.template cast<T>();
    } else {
        return static_cast<T>(OC_FORWARD(arg));
    }
}

template<typename T, typename Arg>
[[nodiscard]] inline auto as(Arg &&arg) {
    if constexpr (is_dsl_v<Arg>) {
        return arg.template as<T>();
    } else {
        return bit_cast<T>(OC_FORWARD(arg));
    }
}

namespace detail {

class ScopeStmtBuilder {
private:
    string _str{};

public:
    ScopeStmtBuilder(const string &str) : _str(str) {}
    template<typename Body>
    void operator+(Body &&body) noexcept {
        comment("start " + _str);
        auto scope = Function::current()->scope();
        Function::current()->with(scope, OC_FORWARD(body));
        comment("end " + _str);
    }
};

class IfStmtBuilder {
private:
    IfStmt *_if{nullptr};

public:
    IfStmtBuilder() = default;
    explicit IfStmtBuilder(IfStmt *stmt) : _if(stmt) {}

    template<typename Condition>
    requires concepts::bool_able<expr_value_t<Condition>>
    [[nodiscard]] static IfStmtBuilder create(Condition &&condition) noexcept {
        IfStmtBuilder builder(Function::current()->if_(extract_expression(OC_FORWARD(condition))));
        return builder;
    }

    template<typename Condition>
    requires concepts::bool_able<expr_value_t<Condition>>
    [[nodiscard]] static IfStmtBuilder create_with_source_location(const string &str, Condition &&condition) noexcept {
        comment(str);
        return create(OC_FORWARD(condition));
    }

    template<typename TrueBranch>
    IfStmtBuilder &operator/(TrueBranch &&true_branch) noexcept {
        Function::current()->with(_if->true_branch(), std::forward<TrueBranch>(true_branch));
        return *this;
    }

    template<typename Func>
    IfStmtBuilder operator*(Func &&func) noexcept {
        return Function::current()->with(_if->false_branch(), std::forward<Func>(func));
    }

    template<typename FalseBranch>
    void operator%(FalseBranch &&false_branch) noexcept {
        Function::current()->with(_if->false_branch(), std::forward<FalseBranch>(false_branch));
    }

    template<typename Condition, typename TrueBranch>
    IfStmtBuilder elif_(Condition &&condition, TrueBranch &&true_branch) noexcept {
        return (*this) * [&] {
            return detail::IfStmtBuilder::create(std::forward<Condition>(condition));
        } / std::forward<TrueBranch>(true_branch);
    }

    template<typename Condition, typename TrueBranch>
    IfStmtBuilder elif_with_source_location(const string &str, Condition &&condition, TrueBranch &&true_branch) noexcept {
        return (*this) * [&] {
            return detail::IfStmtBuilder::create_with_source_location(str, std::forward<Condition>(condition));
        } / std::forward<TrueBranch>(true_branch);
    }

    template<typename FalseBranch>
    void else_(FalseBranch &&false_branch) noexcept {
        (*this) % std::forward<FalseBranch>(false_branch);
    }

    template<typename FalseBranch>
    void else_with_source_location(const string &str, FalseBranch &&false_branch) noexcept {
        comment(str);
        else_(OC_FORWARD(false_branch));
    }
};
}// namespace detail

template<typename Condition, typename TrueBranch>
detail::IfStmtBuilder if_(Condition &&condition,
                          TrueBranch &&true_branch) noexcept {
    return detail::IfStmtBuilder::create(std::forward<Condition>(condition)) / std::forward<TrueBranch>(true_branch);
}

namespace detail {

class CaseStmtBuilder {
private:
    SwitchCaseStmt *_case_stmt{nullptr};

public:
    explicit CaseStmtBuilder(SwitchCaseStmt *stmt)
        : _case_stmt(stmt) {}

    template<typename CaseExpr>
    requires concepts::integral<CaseExpr>
    [[nodiscard]] static CaseStmtBuilder create(CaseExpr &&case_expr) noexcept {
        CaseStmtBuilder builder(Function::current()->switch_case(extract_expression(std::forward<CaseExpr>(case_expr))));
        return builder;
    }

    template<typename CaseExpr>
    requires concepts::integral<CaseExpr>
    [[nodiscard]] static CaseStmtBuilder create_with_source_location(const string &str, CaseExpr &&case_expr) noexcept {
        comment(str);
        return create(OC_FORWARD(case_expr));
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

    DefaultStmtBuilder(const string &str) noexcept
        : _default_stmt(Function::current()->switch_default()) {
        comment(str);
    }

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

    template<typename T>
    requires concepts::switch_able<expr_value_t<T>>
    [[nodiscard]] static SwitchStmtBuilder create_with_source_location(const string &str, T &&t) noexcept {
        comment(str);
        return create(OC_FORWARD(t));
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
    SwitchStmtBuilder &operator*(Body &&func) && noexcept {
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

inline void break_(const string &str = "") noexcept {
    if (!str.empty()) {
        comment(str);
    }
    Function::current()->break_();
}

template<typename Body>
void default_(Body &&body) noexcept {
    detail::DefaultStmtBuilder() * std::forward<Body>(body);
}

inline void continue_(const string &str = "") noexcept {
    if (!str.empty()) {
        comment(str);
    }
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

    static auto create_with_source_location(const string &str) noexcept {
        comment(str);
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
    Var<T> _end;
    Var<T> _step;
    ForStmt *_for_stmt;

public:
    ForStmtBuilder(const Var<T> &begin, const Var<T> &end, const Var<T> &step)
        : _var(begin),
          _end(end),
          _step(step) {
        const BinaryExpr *negative_step = Function::current()->binary(Type::of<bool>(), BinaryOp::LESS, _step.expression(), OC_EXPR(T(0)));
        const BinaryExpr *condition = Function::current()->binary(Type::of<bool>(), BinaryOp::LESS,
                                                            _var.expression(), _end.expression());
        condition = Function::current()->binary(Type::of<bool>(), BinaryOp::BIT_XOR, negative_step, condition);
        _for_stmt = Function::current()->for_(_var.expression(),
                                              condition,
                                              _step.expression());
    }

    template<typename Body>
    void operator/(Body &&body) noexcept {
        Function::current()->with(_for_stmt->body(), [&]() noexcept {
            body(_var);
        });
    }
};
}// namespace detail

template<typename... Args, EPort p = port_v<Args...>>
void print(ocarina::string f, Args &&...args) {
    if constexpr (p == D) {
        size_t num = sizeof...(Args);
        OC_ASSERT(num == substr_count(f, "{}"));
        Function::current()->print(f, vector<const Expression *>{OC_EXPR(args)...});
    } else {
        f += "\n";
        cout << format(f.c_str(), OC_FORWARD(args)...);
    }
}

namespace detail {
template<typename T>
[[nodiscard]] auto to_tuple(const T &t) {
    if constexpr (is_vector_expr_v<T>) {
        using elm_ty = decltype(T::x);
        static constexpr auto dim = vector_expr_dimension_v<T>;
        if constexpr (dim == 2) {
            return std::tuple<elm_ty, elm_ty>(t.x, t.y);
        } else if constexpr (dim == 3) {
            return std::tuple<elm_ty, elm_ty, elm_ty>(t.x, t.y, t.z);
        } else {
            return std::tuple<elm_ty, elm_ty, elm_ty, elm_ty>(t.x, t.y, t.z, t.w);
        }
    } else if constexpr (is_scalar_expr_v<T>) {
        return std::tuple<T>(t);
    }
}
}// namespace detail

template<typename... Args>
void prints(ocarina::string f, Args &&...args) {
    auto all_tuple = std::tuple_cat(detail::to_tuple(args)...);
    auto func = [&]<typename... A, size_t... i>(std::tuple<A...> &&tp, std::index_sequence<i...>) {
        print(f, std::get<i>(OC_FORWARD(tp))...);
    };
    func(ocarina::move(all_tuple), std::make_index_sequence<std::tuple_size_v<decltype(all_tuple)>>());
}

namespace detail {
template<typename Count>
requires concepts::integral<expr_value_t<Count>>
auto range(Count &&count) noexcept {
    return detail::ForStmtBuilder<expr_value_t<Count>>(0, make_expr(std::forward<Count>(count)), 1);
}

template<typename Begin, typename End>
requires concepts::integral<expr_value_t<Begin>>
auto range(Begin &&begin, End &&end) noexcept {
    return detail::ForStmtBuilder<expr_value_t<End>>(make_expr(std::forward<Begin>(begin)),
                                                     make_expr(std::forward<End>(end)),
                                                     1);
}

template<typename Begin, typename End, typename Step>
requires concepts::integral<expr_value_t<Begin>>
auto range(Begin &&begin, End &&end, Step &&step) noexcept {
    return detail::ForStmtBuilder<expr_value_t<End>>(make_expr(std::forward<Begin>(begin)),
                                                     make_expr(std::forward<End>(end)),
                                                     make_expr(std::forward<Step>(step)));
}

template<typename Count>
requires concepts::integral<expr_value_t<Count>>
auto range_with_source_location(const string &str, Count &&count) noexcept {
    comment(str);
    return detail::ForStmtBuilder<expr_value_t<Count>>(0, make_expr(std::forward<Count>(count)), 1);
}

template<typename Begin, typename End>
requires concepts::integral<expr_value_t<Begin>>
auto range_with_source_location(const string &str, Begin &&begin, End &&end) noexcept {
    comment(str);
    return detail::ForStmtBuilder<expr_value_t<End>>(make_expr(std::forward<Begin>(begin)),
                                                     make_expr(std::forward<End>(end)),
                                                     1);
}

template<typename Begin, typename End, typename Step>
requires concepts::integral<expr_value_t<Begin>>
auto range_with_source_location(const string &str, Begin &&begin, End &&end, Step &&step) noexcept {
    comment(str);
    return detail::ForStmtBuilder<expr_value_t<End>>(make_expr(std::forward<Begin>(begin)),
                                                     make_expr(std::forward<End>(end)),
                                                     make_expr(std::forward<Step>(step)));
}

}// namespace detail

template<typename Count, typename Body>
requires concepts::integral<expr_value_t<Count>>
void for_range(Count &&count, Body &&body) noexcept {
    detail::range(std::forward<Count>(count)) / std::forward<Body>(body);
}

template<typename Begin, typename End, typename Body>
requires concepts::integral<expr_value_t<Begin>>
void for_range(Begin &&begin, End &&end, Body &&body) noexcept {
    detail::range(std::forward<Begin>(begin),
                  std::forward<End>(end)) /
        std::forward<Body>(body);
}

template<typename Begin, typename End, typename Step, typename Body>
requires concepts::integral<expr_value_t<Begin>>
void for_range(Begin &&begin, End &&end, Step &&step, Body &&body) noexcept {
    detail::range(std::forward<Begin>(begin),
                  std::forward<End>(end),
                  std::forward<Step>(step)) /
        std::forward<Body>(body);
}

inline void include(string_view fn) noexcept {
    Function::current()->add_header(fn);
}

template<typename... Args>
void configure_block(Args &&...args) noexcept {
    Function::current()->set_block_dim(OC_FORWARD(args)...);
}

template<typename... Args>
void configure_grid(Args &&...args) noexcept {
    Function::current()->set_grid_dim(OC_FORWARD(args)...);
}

template<typename... Args>
void configure(Args &&...args) noexcept {
    Function::current()->configure(OC_FORWARD(args)...);
}

template<typename T>
void return_(T &&ret) noexcept {
    Function::current()->return_(OC_EXPR(ret));
}

inline void return_() noexcept {
    Function::current()->return_(nullptr);
}

}// namespace ocarina