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
struct definition_to_prototype {
    static_assert(always_false_v<T>, "Invalid type in function definition.");
};

template<typename T>
struct definition_to_prototype<Var<T>> {
    using type = T;
};

template<typename T>
struct definition_to_prototype<const Var<T> &> {
    using type = T;
};

template<typename T>
struct definition_to_prototype<Var<T> &> {
    using type = T &;
};

template<typename T>
using definition_to_prototype_t = typename definition_to_prototype<T>::type;

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

template<typename T>
using prototype_to_creation_tag_t = typename detail::prototype_to_creation_tag<T>::type;
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

namespace detail {

template<typename VarTuple, typename TagTuple, typename T, size_t... i>
[[nodiscard]] auto create_argument_definition_impl(T tuple, katana::index_sequence<i...>) {
    (katana::tuple_element_t<i, VarTuple>{katana::tuple_element_t<i, TagTuple>{}}, ...);
}

}// namespace detail

template<typename VarTuple, typename TagTuple>
[[nodiscard]] auto create_argument_definition() {
    return detail::create_argument_definition_impl<VarTuple, TagTuple>(katana::tuple(), katana::make_index_sequence<katana::tuple_size_v<VarTuple>>());
}

template<typename Ret, typename... Args>
class Callable<Ret(Args...)> {
    static_assert(std::negation_v<std::disjunction<std::is_pointer<Args>...>>);

private:
    katana::shared_ptr<const FunctionBuilder> _builder;

    template<typename Func, size_t... i>
    auto init(Func &&func, katana::index_sequence<i...>) {
        cout << typeid(katana::tuple<detail::prototype_to_var_t<Args>...>).name();
        static_assert(std::is_invocable_v<Func, detail::prototype_to_var_t<Args>...>);
        using arg_tuple = katana::tuple<Args...>;
        using var_tuple = katana::tuple<Var<std::remove_cvref_t<Args>>...>;
        using tag_tuple = katana::tuple<detail::prototype_to_creation_tag_t<Args>...>;

        create_argument_definition<var_tuple, tag_tuple>();
        return func(detail::prototype_to_var_t<Args>()...);
    }

public:
    template<typename Func>
    Callable(Func &&func) noexcept
        : _builder(FunctionBuilder::define_callable([&] {
              init(func, katana::index_sequence_for<Args...>());
          })) {}
    //    template<typename Func>
    //    requires std:
};

namespace detail {
template<typename R, typename... Args>
using function_signature = R(Args...);

template<typename T>
struct canonical_signature;

template<typename Ret, typename... Args>
struct canonical_signature<Ret(Args...)> {
    using type = function_signature<Ret, Args...>;
};

template<typename Ret, typename... Args>
struct canonical_signature<Ret (*)(Args...)>
    : canonical_signature<Ret(Args...)> {};

template<typename F>
struct canonical_signature
    : canonical_signature<decltype(&F::operator())> {};

#define KTN_MAKE_MEMBER_FUNC_SIGNATURE(...)                       \
    template<typename Ret, typename Cls, typename... Args>        \
    struct canonical_signature<Ret (Cls::*)(Args...) __VA_ARGS__> \
        : canonical_signature<Ret(Args...)> {};

KTN_MAKE_MEMBER_FUNC_SIGNATURE()
KTN_MAKE_MEMBER_FUNC_SIGNATURE(const)
KTN_MAKE_MEMBER_FUNC_SIGNATURE(volatile)
KTN_MAKE_MEMBER_FUNC_SIGNATURE(noexcept)
KTN_MAKE_MEMBER_FUNC_SIGNATURE(const noexcept)
KTN_MAKE_MEMBER_FUNC_SIGNATURE(const volatile)
KTN_MAKE_MEMBER_FUNC_SIGNATURE(volatile noexcept)
KTN_MAKE_MEMBER_FUNC_SIGNATURE(const volatile noexcept)

#undef KTN_MAKE_MEMBER_FUNC_SIGNATURE

template<typename T>
using canonical_signature_t = typename canonical_signature<T>::type;

template<typename T>
struct dsl_function {
    using type = typename dsl_function<
        canonical_signature_t<std::remove_cvref_t<T>>>::type;
};

template<typename Ret, typename... Args>
struct dsl_function<function_signature<Ret, Args...>> {
    using type = function_signature<
        expr_value_t<Ret>,
        definition_to_prototype_t<Args>...>;
};

template<typename T>
struct dsl_function<Callable<T>> {
    using type = T;
};

template<typename T>
using dsl_function_t = typename dsl_function<T>::type;

}// namespace detail

template<typename T>
Callable(T &&) -> Callable<detail::dsl_function_t<std::remove_cvref_t<T>>>;

}// namespace katana