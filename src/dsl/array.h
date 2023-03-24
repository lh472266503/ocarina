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
    uint _size{};
    const Expression *_expression{};

public:
    explicit Array(size_t num, const Expression *expression = nullptr)
        : _size(num),
          _expression(expression == nullptr ? Function::current()->local(type()) : expression) {}

    template<typename U>
    requires is_vector_v<U> && concepts::different<std::remove_cvref_t<U>, Array<T>>
    explicit Array(U &&vec) noexcept
        : Array(vector_expr_dimension_v<U>) {
        for (int i = 0; i < size(); ++i) {
            (*this)[i] = vec[i];
        }
    }

    Array(const Array &other) noexcept
        : _size{other._size}, _expression(Function::current()->local(type())) {
        Function::current()->assign(_expression, other._expression);
    }

    Array(Array &&) noexcept = default;

    [[nodiscard]] static const Type *type(uint size) noexcept {
        return Type::from(ocarina::format("array<{},{}>",
                                          detail::TypeDesc<T>::description(),
                                          size));
    }

    [[nodiscard]] const Type *type() const noexcept {
        return type(_size);
    }

    template<typename U>
    requires is_array_expr_v<U>
    static Array<T> create(U &&array) noexcept {
        Array<T> ret{array.size()};
        for (uint i = 0; i < ret.size(); ++i) {
            ret[i] = array[i];
        }
        return ret;
    }

    template<typename... Args>
    static Array<T> create(Args &&...args) noexcept {
        return create(std::array<Var<T>, sizeof...(args)>{OC_FORWARD(args)...});
    }

    [[nodiscard]] Var<T> to_scalar() const noexcept {
        return (*this)[0];
    }

    template<size_t N>
    [[nodiscard]] Var<Vector<T, N>> to_vec() const noexcept {
        OC_ASSERT(N <= _size);
        Var<Vector<T, N>> ret;
        for (int i = 0; i < N; ++i) {
            ret[i] = (*this)[i];
        }
        return ret;
    }

    [[nodiscard]] Var<Vector<T, 2>> to_vec2() const noexcept { return to_vec<2>(); }
    [[nodiscard]] Var<Vector<T, 3>> to_vec3() const noexcept { return to_vec<3>(); }
    [[nodiscard]] Var<Vector<T, 4>> to_vec4() const noexcept { return to_vec<4>(); }

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

    [[nodiscard]] const Expression *expression() const noexcept { return _expression; }
    [[nodiscard]] uint size() const noexcept { return _size; }

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

namespace detail {

template<typename T>
template<typename U, typename V>
requires(is_all_floating_point_expr_v<U, V>)
    Array<float> EnableTextureSample<T>::sample(uint channel_num, const U &u, const V &v)
const noexcept {
    const T *texture = static_cast<const T *>(this);
    const CallExpr *expr = Function::current()->call_builtin(Array<float>::type(channel_num),
                                                             CallOp::TEX_SAMPLE,
                                                             {texture->expression(),
                                                              OC_EXPR(u),
                                                              OC_EXPR(v)},
                                                             {channel_num});
    return Array<float>(channel_num, expr);
}

template<typename T>
template<typename U, typename V, typename W>
requires(is_all_floating_point_expr_v<U, V, W>)
    Array<float> EnableTextureSample<T>::sample(uint channel_num, const U &u, const V &v, const W &w)
const noexcept {
    const T *texture = static_cast<const T *>(this);
    const CallExpr *expr = Function::current()->call_builtin(Array<float>::type(channel_num),
                                                             CallOp::TEX_SAMPLE,
                                                             {texture->expression(),
                                                              OC_EXPR(u),
                                                              OC_EXPR(v),
                                                              OC_EXPR(w)},
                                                             {channel_num});
    return Array<float>(channel_num, expr);
}

template<typename T>
template<typename UVW>
requires(is_float_vector3_v<expr_value_t<UVW>>)
    Array<float> EnableTextureSample<T>::sample(uint channel_num, const UVW &uvw)
const noexcept {
    return sample(channel_num, uvw.x, uvw.y, uvw.z);
}

template<typename T>
template<typename UV>
requires(is_float_vector2_v<expr_value_t<UV>>)
    Array<float> EnableTextureSample<T>::sample(uint channel_num, const UV &uv)
const noexcept {
    return sample(channel_num, uv.x, uv.y);
}

}// namespace detail

}// namespace ocarina