//
// Created by Zero on 15/05/2022.
//

#pragma once

#include "ast/type_registry.h"
#include "dsl/var.h"

#define OC_MAKE_STRUCT_MEMBER(member) \
    Var<std::remove_cvref_t<decltype(this_type::member)>> member{};

#define OC_MAKE_COMPUTABLE_BODY(S, ...)                                        \
    namespace detail {                                                         \
    template<>                                                                 \
    struct Computable<S> {                                                     \
        using this_type = S;                                                   \
        MAP(OC_MAKE_STRUCT_MEMBER, ##__VA_ARGS__)                              \
    private:                                                                   \
        const Expression *_expression{nullptr};                                \
                                                                               \
    public:                                                                    \
        [[nodiscard]] auto expression() const noexcept { return _expression; } \
                                                                               \
    protected:                                                                 \
        explicit Computable(const Expression *e) noexcept : _expression{e} {}  \
        Computable(Computable &&) noexcept = default;                          \
        Computable(const Computable &) noexcept = default;                     \
    };                                                                         \
    }
#define OC_MAKE_VAR_BODY(S, ...)                                                                \
    template<>                                                                                  \
    struct Var<S> : public detail::Computable<S> {                                              \
        MAP(OC_MAKE_STRUCT_MEMBER, ##__VA_ARGS__)                                               \
        explicit Var(const Expression *expression) noexcept                                     \
            : detail::Computable<S>(expression) {}                                              \
                                                                                                \
        Var() noexcept : Var(Function::current()->local(Type::of<S>())) {}                      \
                                                                                                \
        template<typename Arg>                                                                  \
        requires concepts::non_pointer<std::remove_cvref_t<Arg>> &&                             \
            concepts::assign_able<expr_value_t<std::remove_cvref_t<S>>, expr_value_t<Arg>>      \
            Var(Arg &&arg) : Var() {                                                            \
            assign(*this, std::forward<Arg>(arg));                                              \
        }                                                                                       \
        explicit Var(detail::ArgumentCreation) noexcept                                         \
            : Var(Function::current()->argument(Type::of<S>())) {                               \
        }                                                                                       \
        explicit Var(detail::ReferenceArgumentCreation) noexcept                                \
            : Var(Function::current()->reference_argument(Type::of<S>())) {}                    \
        template<typename Arg>                                                                  \
        requires concepts::assign_able<expr_value_t<std::remove_cvref_t<S>>, expr_value_t<Arg>> \
        void operator=(Arg &&arg) {                                                             \
            assign(*this, std::forward<Arg>(arg));                                              \
        }                                                                                       \
    };

#define OC_STRUCT(S, ...)                       \
    OC_MAKE_STRUCT_REFLECTION(S, ##__VA_ARGS__) \
    OC_MAKE_STRUCT_DESC(S, ##__VA_ARGS__)       \
    OC_MAKE_COMPUTABLE_BODY(S, ##__VA_ARGS__) \
    OC_MAKE_VAR_BODY(S, ##__VA_ARGS__) \
