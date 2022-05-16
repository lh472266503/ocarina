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
#include "arg.h"

namespace katana {

namespace detail {
template<typename T>
struct prototype_to_creation_tag {
    using type = ArgumentCreation;
};

template<typename T>
struct prototype_to_creation_tag<const T &> {
    using type = ArgumentCreation;
};

template<typename T>
struct prototype_to_creation_tag<T &> {
    using type = ReferenceArgumentCreation;
};
}// namespace detail

template<typename T>
using prototype_to_creation_tag_t = typename detail::prototype_to_creation_tag<T>::type;

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

namespace detail {

template<typename VarTuple, typename TagTuple, typename T, size_t... i>
[[nodiscard]] auto create_argument_tuple_impl(T tuple, std::index_sequence<i...>) {
}

}// namespace detail

template<typename VarTuple, typename TagTuple, typename T>
[[nodiscard]] auto create_argument_tuple(T tuple) {
    return detail::create_argument_tuple_impl<VarTuple, TagTuple>(tuple, std::make_index_sequence<std::tuple_size_v<VarTuple>>());
}

template<typename Ret, typename... Args>
class Callable<Ret(Args...)> {
    static_assert(std::negation_v<std::disjunction<std::is_pointer<Args>...>>);

private:
    katana::shared_ptr<FunctionBuilder> _builder;

public:
    template<typename Func>
    explicit Callable(Func &&func) noexcept
        : _builder(FunctionBuilder::define_callable([&] {
              static_assert(std::is_invocable_v<Func, detail::prototype_to_var<Args>...>);
              using arg_tuple = std::tuple<Args...>;
              using var_tuple = std::tuple<Var<std::remove_cvref_t<Args>>...>;
              using tag_tuple = std::tuple<prototype_to_creation_tag_t<Args>...>;

              auto args = create_argument_tuple<var_tuple, tag_tuple>(std::tuple());
          })) {
    }
    //    template<typename Func>
    //    requires std:
};

}// namespace katana