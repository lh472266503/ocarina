//
// Created by Zero on 17/07/2022.
//

#pragma once

#include "core/stl.h"
#include "computable.h"
#include "ast/type_registry.h"
#include "core/string_util.h"

namespace ocarina {

template<typename T>
class Array : public detail::EnableSubscriptAccess<Array<T>, T> {
public:
    using element_type = T;

private:
    size_t _size{};
    const RefExpr *_expression{};

public:
    explicit Array(size_t num)
        : _size(num) {
        const Type *type = Type::from(ocarina::format("array<{},{}>",
                                                      detail::TypeDesc<T>::description(),
                                                      num));
        _expression = Function::current()->local(type);
    }

    template<typename U>
    requires is_vector_v<U> && concepts::different<std::remove_cvref_t<U>, Array<T>>
    explicit Array(U &&vec) noexcept
        : Array(vector_expr_dimension_v<U>) {
        for (int i = 0; i < size(); ++i) {
            (*this)[i] = vec[i];
        }
    }

    Array(const Array &other) noexcept
        : _size{other._size} {
        const Type *type = Type::from(ocarina::format("array<{},{}>",
                                                      detail::TypeDesc<T>::description(),
                                                      _size));
        _expression = Function::current()->local(type);
        Function::current()->assign(_expression, other._expression);
    }

    Array(Array &&) noexcept = default;

    template<typename U>
    requires is_array_expr_v<U>
    static Array<T> create(U &&array) noexcept {
        Array<T> ret{array.size()};
        for (uint i = 0; i < ret.size(); ++i) {
            ret[i] = array[i];
        }
        return ret;
    }

    template<typename...Args>
    static Array<T> create(Args &&...args) noexcept {
        return create(std::array<Var<T>, sizeof...(args)>{OC_FORWARD(args)...});
    }

    [[nodiscard]] Var<T> to_scalar() const noexcept {
        return (*this)[0];
    }

    [[nodiscard]] Var<Vector<T, 2>> to_vec2() const noexcept {
        OC_ASSERT(2 <= _size);
        Var<Vector<T, 2>> ret;
        ret.x = (*this)[0];
        ret.y = (*this)[1];
        return ret;
    }

    [[nodiscard]] Var<Vector<T, 3>> to_vec3() const noexcept {
        OC_ASSERT(3 <= _size);
        Var<Vector<T, 3>> ret;
        ret.x = (*this)[0];
        ret.y = (*this)[1];
        ret.z = (*this)[2];
        return ret;
    }

    [[nodiscard]] Var<Vector<T, 4>> to_vec4() const noexcept {
        OC_ASSERT(4 <= _size);
        Var<Vector<T, 4>> ret;
        ret.x = (*this)[0];
        ret.y = (*this)[1];
        ret.z = (*this)[2];
        ret.w = (*this)[3];
        return ret;
    }

    Array &operator=(const Array &rhs) noexcept {
        if (&rhs != this) [[likely]] {
            OC_ASSERT(_size == rhs._size);
            Function::current()->assign(_expression, rhs._expression);
        }
        return *this;
    }
    Array &operator=(Array &&rhs) noexcept {
        *this = static_cast<const Array &>(rhs);
        return *this;
    }

    [[nodiscard]] const RefExpr *expression() const noexcept { return _expression; }
    [[nodiscard]] size_t size() const noexcept { return _size; }

    template<typename F>
    [[nodiscard]] Array<T> map(F &&f) const noexcept {
        Array<T> s{size()};
        for (auto i = 0u; i < size(); i++) {
            if constexpr (std::invocable<F, Var<T>>) {
                s[i] = f((*this)[i]);
            } else {
                s[i] = f(i, (*this)[i]);
            }
        }
        return s;
    }

    template<typename T, typename F>
    [[nodiscard]] auto reduce(T &&initial, F &&f) const noexcept {
        auto r = eval(OC_FORWARD(initial));
        for (auto i = 0u; i < size(); i++) {
            if constexpr (std::invocable<F, Var<expr_value_t<decltype(r)>>, Float>) {
                r = f(r, (*this)[i]);
            } else {
                r = f(r, i, (*this)[i]);
            }
        }
        return r;
    }
};

}// namespace ocarina