//
// Created by Zero on 2024/2/25.
//

#pragma once

#include "core/basic_types.h"
#include "builtin.h"
#include "var.h"

namespace ocarina {

template<typename T>
struct SOAView {
    static_assert(always_false_v<T>, "The SOAView template must be specialized");
};

#define OC_MAKE_ATOMIC_SOA(TemplateArgs, TypeName)                                          \
    TemplateArgs struct SOAView<TypeName> {                                                 \
    public:                                                                                 \
        using element_type = TypeName;                                                      \
        static constexpr uint type_size = sizeof(element_type);                             \
                                                                                            \
    private:                                                                                \
        ByteBufferVar *_buffer{};                                                           \
        Uint _view_size{};                                                                  \
        Uint _offset{};                                                                     \
        uint _stride{};                                                                     \
                                                                                            \
    public:                                                                                 \
        SOAView() = default;                                                                \
        SOAView(ByteBufferVar &buffer, const Uint &view_size,                               \
                const Uint &ofs, uint stride)                                               \
            : _buffer(&buffer), _view_size(view_size),                                      \
              _offset(ofs), _stride(stride) {}                                              \
                                                                                            \
        template<typename Index>                                                            \
        requires is_integral_expr_v<Index>                                                  \
        [[nodiscard]] Var<element_type> read(Index &&index) const noexcept {                \
            return _buffer->load_as<element_type>(_offset + OC_FORWARD(index) * type_size); \
        }                                                                                   \
                                                                                            \
        template<typename Index>                                                            \
        requires is_integral_expr_v<Index>                                                  \
        void write(Index &&index, const Var<element_type> &val) noexcept {                  \
            _buffer->store(_offset + OC_FORWARD(index) * type_size, val);                   \
        }                                                                                   \
                                                                                            \
        template<typename int_type = uint>                                                  \
        [[nodiscard]] Var<int_type> size_in_byte() const noexcept {                         \
            return _view_size / _stride * type_size;                                        \
        }                                                                                   \
    };

#define OC_COMMA ,

OC_MAKE_ATOMIC_SOA(template<>, uint)
OC_MAKE_ATOMIC_SOA(template<>, uint64t)
OC_MAKE_ATOMIC_SOA(template<>, float)
OC_MAKE_ATOMIC_SOA(template<>, int)
OC_MAKE_ATOMIC_SOA(template<typename T OC_COMMA uint N>,
                   array<T OC_COMMA N>)

}// namespace ocarina

#define OC_MAKE_SOA_MEMBER(field_name) ocarina::SOAView<decltype(element_type::field_name)> field_name;

#define OC_MAKE_SOA_MEMBER_CONSTRUCT(field_name)                                                              \
    field_name = ocarina::SOAView<decltype(element_type::field_name)>(buffer_var, view_size, offset, stride); \
    offset += field_name.size_in_byte();

#define OC_MAKE_SOA_MEMBER_READ(field_name) ret.field_name = field_name.read(OC_FORWARD(index));

#define OC_MAKE_SOA_MEMBER_WRITE(field_name) field_name.write(OC_FORWARD(index), val.field_name);

#define OC_MAKE_SOA_MEMBER_SIZE(field_name) ret += field_name.size_in_byte();

#define OC_MAKE_STRUCT_SOA_VIEW(TemplateArgs, S, ...)

namespace ocarina {

template<typename T>
struct SOAView<Vector<T, 2>> {
public:
    using element_type = Vector<T, 2>;
    static constexpr uint type_size = sizeof(element_type);

public:
    SOAView<T> x;
    SOAView<T> y;

public:
    SOAView() = default;
    explicit SOAView(ByteBufferVar &buffer_var,
                     Uint view_size = InvalidUI32,
                     Uint offset = 0u,
                     uint stride = type_size) {
        view_size = ocarina::min(buffer_var.size<uint>(), view_size);
        OC_MAKE_SOA_MEMBER_CONSTRUCT(x)
        OC_MAKE_SOA_MEMBER_CONSTRUCT(y)
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<element_type> read(Index &&index) const noexcept {
        Var<element_type> ret;
        ret.x = x.read(OC_FORWARD(index));
        ret.y = y.read(OC_FORWARD(index));
        return ret;
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    void write(Index &&index, const Var<element_type> &val) noexcept {
        x.write(OC_FORWARD(index), val.x);
        y.write(OC_FORWARD(index), val.y);
    }

    template<typename int_type = uint>
    [[nodiscard]] Var<int_type> size_in_byte() const noexcept {
        Var<int_type> ret = 0;
        ret += x.size_in_byte();
        ret += y.size_in_byte();
        return ret;
    }
};
}// namespace ocarina

namespace ocarina {

template<typename T>
struct SOAView<Vector<T, 3>> {
public:
    using element_type = Vector<T, 3>;
    static constexpr uint type_size = sizeof(element_type);

public:
    SOAView<T> x;
    SOAView<T> y;
    SOAView<T> z;

public:
    SOAView() = default;
    explicit SOAView(ByteBufferVar &buffer_var,
                     Uint view_size = InvalidUI32,
                     Uint offset = 0u,
                     uint stride = type_size) {
        view_size = ocarina::min(buffer_var.size<uint>(), view_size);
        OC_MAKE_SOA_MEMBER_CONSTRUCT(x)
        OC_MAKE_SOA_MEMBER_CONSTRUCT(y)
        OC_MAKE_SOA_MEMBER_CONSTRUCT(z)
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<element_type> read(Index &&index) const noexcept {
        Var<element_type> ret;
        ret.x = x.read(OC_FORWARD(index));
        ret.y = y.read(OC_FORWARD(index));
        ret.z = z.read(OC_FORWARD(index));
        return ret;
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    void write(Index &&index, const Var<element_type> &val) noexcept {
        x.write(OC_FORWARD(index), val.x);
        y.write(OC_FORWARD(index), val.y);
        z.write(OC_FORWARD(index), val.z);
    }

    template<typename int_type = uint>
    [[nodiscard]] Var<int_type> size_in_byte() const noexcept {
        Var<int_type> ret = 0;
        ret += x.size_in_byte();
        ret += y.size_in_byte();
        ret += z.size_in_byte();
        return ret;
    }
};
}// namespace ocarina

namespace ocarina {

template<typename T>
struct SOAView<Vector<T, 4>> {
public:
    using element_type = Vector<T, 4>;
    static constexpr uint type_size = sizeof(element_type);

public:
    OC_MAKE_SOA_MEMBER(x)
    SOAView<T> y;
    SOAView<T> z;
    SOAView<T> w;

public:
    SOAView() = default;
    explicit SOAView(ByteBufferVar &buffer_var,
                     Uint view_size = InvalidUI32,
                     Uint offset = 0u,
                     uint stride = type_size) {
        view_size = ocarina::min(buffer_var.size<uint>(), view_size);
        OC_MAKE_SOA_MEMBER_CONSTRUCT(x)
        OC_MAKE_SOA_MEMBER_CONSTRUCT(y)
        OC_MAKE_SOA_MEMBER_CONSTRUCT(z)
        OC_MAKE_SOA_MEMBER_CONSTRUCT(w)
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<element_type> read(Index &&index) const noexcept {
        Var<element_type> ret;
        OC_MAKE_SOA_MEMBER_READ(x)
        ret.y = y.read(OC_FORWARD(index));
        ret.z = z.read(OC_FORWARD(index));
        ret.w = w.read(OC_FORWARD(index));
        return ret;
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    void write(Index &&index, const Var<element_type> &val) noexcept {
        OC_MAKE_SOA_MEMBER_WRITE(x)
        y.write(OC_FORWARD(index), val.y);
        z.write(OC_FORWARD(index), val.z);
        w.write(OC_FORWARD(index), val.w);
    }

    template<typename int_type = uint>
    [[nodiscard]] Var<int_type> size_in_byte() const noexcept {
        Var<int_type> ret = 0;
        OC_MAKE_SOA_MEMBER_SIZE(x)
        ret += y.size_in_byte();
        ret += z.size_in_byte();
        ret += w.size_in_byte();
        return ret;
    }
};

}// namespace ocarina

namespace ocarina {

template<uint N>
struct SOAView<Matrix<N>> {
public:
    using element_type = Matrix<N>;
    static constexpr uint type_size = sizeof(element_type);

private:
    array<SOAView<Vector<float, N>>, N> _cols{};

public:
    explicit SOAView(ByteBufferVar &buffer_var,
                     Uint view_size = InvalidUI32,
                     Uint offset = 0u,
                     uint stride = type_size) {
        view_size = ocarina::min(buffer_var.size<uint>(), view_size);
        for (int i = 0; i < N; ++i) {
            _cols[i] = SOAView<Vector<float, N>>(buffer_var, view_size,
                                                 offset, stride);
            offset += _cols[i].size_in_byte();
        }
    }

    [[nodiscard]] auto &operator[](size_t index) const noexcept { return _cols[index]; }
    [[nodiscard]] auto &operator[](size_t index) noexcept { return _cols[index]; }

    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<element_type> read(Index &&index) const noexcept {
        Var<element_type> ret;
        for (int i = 0; i < N; ++i) {
            ret[i] = _cols[i].read(OC_FORWARD(index));
        }
        return ret;
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    void write(Index &&index, const Var<element_type> &val) noexcept {
        for (int i = 0; i < N; ++i) {
            _cols[i].write(OC_FORWARD(index), val[i]);
        }
    }

    template<typename int_type = uint>
    [[nodiscard]] Var<int_type> size_in_byte() const noexcept {
        Var<int_type> ret = 0;
        for (int i = 0; i < N; ++i) {
            ret += _cols[i].size_in_byte();
        }
        return ret;
    }
};

#undef OC_MAKE_ATOMIC_SOA

}// namespace ocarina