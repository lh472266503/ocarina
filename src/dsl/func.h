//
// Created by Zero on 02/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "var.h"
#include "builtin.h"
#include "expr.h"
#include "ast/function_builder.h"

namespace katana::dsl {

namespace detail {
template<typename T>
struct var_to_prototype {
    static_assert(always_false_v<T>, "Invalid type in function definition.");
};

template<typename T>
struct var_to_prototype<Var<T>> {
    using type = T;
};

template<typename T>
struct var_to_prototype<const Var<T> &> {
    using type = T;
};

template<typename T>
struct var_to_prototype<Var<T> &> {
    using type = T &;
};

template<typename T>
struct prototype_to_callable_invocation {
    using type = Expr<T>;
};

template<typename T>
struct prototype_to_callable_invocation<const T &> {
    using type = Expr<T>;
};


}// namespace detail

template<typename T>
class Callable {
    static_assert(always_false_v<T>);
};

template<typename T>
struct is_kernel : std::false_type {};

template<typename T>
struct is_callable : std::false_type {};

template<typename T>
struct is_callable<Callable<T>> : std::true_type {};

template<typename Ret, typename... Args>
class Callable<Ret(Args...)> {
    static_assert(std::negation_v<std::disjunction<std::is_pointer<Args>...>>);

private:
    katana::shared_ptr<ast::FunctionBuilder> _builder;

public:
//    template<typename Func>
//    requires std:

};

}// namespace katana::dsl