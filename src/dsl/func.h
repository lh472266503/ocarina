//
// Created by Zero on 02/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "var.h"
#include "builtin.h"
#include "expr.h"
#include "syntax.h"
#include "expr_traits.h"
#include "arg.h"

namespace ocarina {

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

#define OC_MAKE_MEMBER_FUNC_SIGNATURE(...)                        \
    template<typename Ret, typename Cls, typename... Args>        \
    struct canonical_signature<Ret (Cls::*)(Args...) __VA_ARGS__> \
        : canonical_signature<Ret(Args...)> {};

OC_MAKE_MEMBER_FUNC_SIGNATURE()
OC_MAKE_MEMBER_FUNC_SIGNATURE(const)
OC_MAKE_MEMBER_FUNC_SIGNATURE(volatile)
OC_MAKE_MEMBER_FUNC_SIGNATURE(noexcept)
OC_MAKE_MEMBER_FUNC_SIGNATURE(const noexcept)
OC_MAKE_MEMBER_FUNC_SIGNATURE(const volatile)
OC_MAKE_MEMBER_FUNC_SIGNATURE(volatile noexcept)
OC_MAKE_MEMBER_FUNC_SIGNATURE(const volatile noexcept)

#undef OC_MAKE_MEMBER_FUNC_SIGNATURE

template<typename T>
using canonical_signature_t = typename canonical_signature<T>::type;

template<typename T>
struct dsl_function {
    using type = typename dsl_function<
        canonical_signature_t<std::remove_cvref_t<T>>>::type;
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

namespace detail {

template<typename T, typename... A>
[[nodiscard]] auto tuple_insert(ocarina::tuple<A...> lst, T &&t) {
    using ret_type = ocarina::tuple<T, A...>;
    auto func = []<typename TT, typename... AA, size_t... i>(TT && t,
                                                           ocarina::tuple<AA...> lst,
                                                           std::index_sequence<i...>)
                    ->ret_type {
        return ret_type(std::forward<TT>(t), std::move(ocarina::get<i>(lst))...);
    };
    return func(std::forward<T>(t), std::move(lst), std::index_sequence_for<A...>());
}

[[nodiscard]] inline ocarina::tuple<> create_argument_definition_impl(ocarina::tuple<> *var_tuple,
                                                                      ocarina::tuple<> *tag_tuple) {
    return {};
}

template<typename Var, typename... RestVar, typename Tag, typename... RestTag>
[[nodiscard]] auto create_argument_definition_impl(ocarina::tuple<Var, RestVar...> *var_tuple,
                                                   ocarina::tuple<Tag, RestTag...> *tag_tuple) {
    return tuple_insert(std::move(create_argument_definition_impl(static_cast<tuple<RestVar...> *>(nullptr),
                                                                  static_cast<tuple<RestTag...> *>(nullptr))),
                        Var{Tag{}});
}

}// namespace detail

template<typename VarTuple, typename TagTuple>
[[nodiscard]] auto create_argument_definition() {
    return detail::create_argument_definition_impl(static_cast<VarTuple *>(nullptr), static_cast<TagTuple *>(nullptr));
}

namespace detail {
template<typename... Args, typename Func, size_t... i>
auto create(Func &&func, ocarina::index_sequence<i...>) {
    static_assert(std::is_invocable_v<Func, detail::prototype_to_var_t<Args>...>);
    using var_tuple = ocarina::tuple<Var<std::remove_cvref_t<Args>>...>;
    using tag_tuple = ocarina::tuple<detail::prototype_to_creation_tag_t<Args>...>;
    auto args = create_argument_definition<var_tuple, tag_tuple>();
    return func(std::forward<detail::prototype_to_var_t<Args>>(ocarina::get<i>(args))...);
}
}// namespace detail

class FuncWrapper : public concepts::Noncopyable {
protected:
    ocarina::unique_ptr<Function> _function;
    explicit FuncWrapper(ocarina::unique_ptr<Function> f) : _function(std::move(f)) {}

public:
    [[nodiscard]] const Function &function() const noexcept { return *_function; }
};

namespace detail {

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

template<typename Ret, typename... Args>
class Callable<Ret(Args...)> : public FuncWrapper {
    static_assert(std::negation_v<std::disjunction<std::is_pointer<Args>...>>);

public:
    using signature = typename detail::canonical_signature_t<Ret(Args...)>;

public:
    template<typename Func>
    Callable(Func &&func) noexcept
        : FuncWrapper(std::move(Function::define_callable([&] {
              if constexpr (std::is_same_v<void, Ret>) {
                  detail::create<Args...>(OC_FORWARD(func), ocarina::index_sequence_for<Args...>());
              } else {
                  decltype(auto) ret = detail::create<Args...>(OC_FORWARD(func), ocarina::index_sequence_for<Args...>());
                  Function::current()->return_(ret.expression());
              }
          }))) {}

    template<typename... A>
    requires std::is_invocable_v<signature, expr_value_t<A>...>
    auto operator()(A &&...args) const noexcept {
        const CallExpr *expr = Function::current()->call(Type::of<Ret>(), _function.get(), {(OC_EXPR(args))...});
        if constexpr (!std::is_same_v<std::remove_cvref_t<Ret>, void>) {
            return def<Ret>(expr);
        } else {
            Function::current()->expr_statement(expr);
        }
    }
};

template<typename T>
Callable(T &&) -> Callable<detail::dsl_function_t<std::remove_cvref_t<T>>>;

template<size_t Dim = 1, typename... Args>
class Kernel : public FuncWrapper {
public:
    static constexpr auto dimension = Dim;
    static_assert(Dim >= 1 && Dim <= 3, "dimension of kernel must be in range[1, 3]!");

public:
    using signature = typename detail::canonical_signature_t<void(Args...)>;

    template<typename Func>
    Kernel(Func &&func) noexcept
        : FuncWrapper(std::move(Function::define_kernel([&] {
              detail::create<Args...>(OC_FORWARD(func), ocarina::index_sequence_for<Args...>());
          }))) {}
};

#define OC_MAKE_KERNEL(dim)                                                               \
    template<typename... Args>                                                            \
    class Kernel##dim##D : public Kernel<dim, Args...> {                                  \
    public:                                                                               \
        template<typename Func>                                                           \
        Kernel##dim##D(Func &&func) noexcept                                              \
            : Kernel<dim, Args...>(std::forward<Func>(func)) {}                           \
    };                                                                                    \
    template<typename T>                                                                  \
    Kernel##dim##D(T &&)->Kernel##dim##D<detail::dsl_function_t<std::remove_cvref_t<T>>>; \
    namespace detail {                                                                    \
    template<typename T>                                                                  \
    struct dsl_function<Kernel##dim##D<T>> {                                              \
        using type = T;                                                                   \
    };                                                                                    \
    }

OC_MAKE_KERNEL(1)
OC_MAKE_KERNEL(2)
OC_MAKE_KERNEL(3)

#undef OC_MAKE_KERNEL

}// namespace ocarina