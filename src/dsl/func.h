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
#include "expr_traits.h"

namespace katana::dsl {

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