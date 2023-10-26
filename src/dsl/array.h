//
// Created by Zero on 17/07/2022.
//

#pragma once

#include "core/stl.h"
#include "computable.h"
#include "ast/type_registry.h"
#include "core/string_util.h"
#include "syntax_sugar.h"

namespace ocarina {

template<typename T>
class Array : public detail::EnableSubscriptAccess<Array<T>, T> {
public:
    using element_type = T;

private:
    uint _size{};
    const Expression *_expression{};

public:
    Array() = default;

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

    explicit Array(const vector<T> &vec) noexcept
        : Array(vec.size()) {
        for (int i = 0; i < size(); ++i) {
            (*this)[i] = vec[i];
        }
    }

    Array(const Array &other) noexcept
        : _size{other._size}, _expression(Function::current()->local(type())) {
        Function::current()->assign(_expression, other._expression);
    }
    Array(Array &&) noexcept = default;

    void reset(const Array<T> &array) noexcept {
        _size = array.size();
        _expression = array.expression();
    }
    void invalidate() noexcept {
        _size = 0;
        _expression = nullptr;
    }

    [[nodiscard]] bool valid() const noexcept { return _size > 0; }

    [[nodiscard]] static const Type *type(uint size) noexcept {
        return Type::from(ocarina::format("array<{},{}>",
                                          TypeDesc<T>::description(),
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

    [[nodiscard]] Var<T> as_scalar() const noexcept {
        return (*this)[0];
    }

    template<size_t N>
    [[nodiscard]] Var<Vector<T, N>> as_vec() const noexcept {
        OC_ASSERT(N <= _size);
        Var<Vector<T, N>> ret;
        for (int i = 0; i < N; ++i) {
            ret[i] = (*this)[i];
        }
        return ret;
    }

    [[nodiscard]] Array<T> sub(uint offset, uint num) const noexcept {
        Array<T> ret{num};
        for (int i = 0; i < num; ++i) {
            ret[i] = at(i + offset);
        }
        return ret;
    }

    [[nodiscard]] Var<Vector<T, 2>> as_vec2() const noexcept { return as_vec<2>(); }
    [[nodiscard]] Var<Vector<T, 3>> as_vec3() const noexcept { return as_vec<3>(); }
    [[nodiscard]] Var<Vector<T, 4>> as_vec4() const noexcept { return as_vec<4>(); }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] decltype(auto) at(Index &&index) const noexcept {
        return (*this)[OC_FORWARD(index)];
    }

#include "swizzle_inl/array_swizzle.inl.h"

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

    template<typename I, typename F>
    [[nodiscard]] auto reduce(I &&initial, F &&f) const noexcept {
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

template<typename T>
class Container : public Array<T> {
private:
    Uint _count{0u};
    using Super = Array<T>;

public:
    using Array<T>::Array;
    void push_back(const Var<T> &t) noexcept {
        Super::operator[](_count) = t;
        _count += 1;
    }
    void pop() noexcept {
        _count -= 1;
    }
    [[nodiscard]] Var<T> top() const noexcept {
        return Super::operator[](_count - 1);
    }
    [[nodiscard]] Uint count() const noexcept {
        return _count;
    }
    void clear() noexcept {
        _count = 0;
    }
    template<typename F>
    void for_each(F &&f) const noexcept {
        $for(i, _count) {
            if constexpr (std::invocable<F, Var<T>>) {
                f((*this)[i]);
            } else {
                f(i, (*this)[i]);
            }
        };
    }
};

namespace detail {

template<typename T>
[[nodiscard]] inline Array<T> eval_array(const Array<T> &array) noexcept {
    Array<T> ret{array.size()};
    ret = array;
    return ret;
}

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
    return eval_array(Array<float>(channel_num, expr));
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
    return eval_array(Array<float>(channel_num, expr));
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

template<typename T, typename Offset>
[[nodiscard]] Array<T> ResourceArrayByteBuffer::read_dynamic_array(uint size, Offset &&offset) const noexcept {
    const CallExpr *expr = Function::current()->call_builtin(Array<T>::type(size),
                                                             CallOp::RESOURCE_ARRAY_BYTE_BUFFER_READ,
                                                             {_resource_array, _index, OC_EXPR(offset)});
    return detail::eval_array(Array<T>(size, expr));
}

template<typename U, typename V>
requires(is_all_floating_point_expr_v<U, V>)
Array<float> ResourceArrayTexture::sample(uint channel_num, const U &u, const V &v)
    const noexcept {
    const CallExpr *expr = Function::current()->call_builtin(Array<float>::type(channel_num),
                                                             CallOp::RESOURCE_ARRAY_TEX_SAMPLE,
                                                             {_resource_array, _index, OC_EXPR(u), OC_EXPR(v)},
                                                             {channel_num});
    return detail::eval_array(Array<float>(channel_num, expr));
}

template<typename U, typename V, typename W>
requires(is_all_floating_point_expr_v<U, V, W>)
Array<float> ResourceArrayTexture::sample(uint channel_num, const U &u, const V &v, const W &w)
    const noexcept {
    const CallExpr *expr = Function::current()->call_builtin(Array<float>::type(channel_num),
                                                             CallOp::RESOURCE_ARRAY_TEX_SAMPLE,
                                                             {_resource_array, _index, OC_EXPR(u), OC_EXPR(v), OC_EXPR(w)},
                                                             {channel_num});
    return detail::eval_array(Array<float>(channel_num, expr));
}

template<typename UVW>
requires(is_float_vector3_v<expr_value_t<UVW>>)
Array<float> ResourceArrayTexture::sample(uint channel_num, const UVW &uvw)
    const noexcept {
    return sample(channel_num, uvw.x, uvw.y, uvw.z);
}

template<typename UV>
requires(is_float_vector2_v<expr_value_t<UV>>)
Array<float> ResourceArrayTexture::sample(uint channel_num, const UV &uv)
    const noexcept {
    return sample(channel_num, uv.x, uv.y);
}

}// namespace ocarina