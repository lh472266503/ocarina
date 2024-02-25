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
    static_assert(always_false_v<T>);
};

#define OC_MAKE_ATOMIC_SOA(TypeName, TemplateArg)                                   \
    template<TemplateArg>                                                           \
    struct SOAView<TypeName> {                                                      \
    public:                                                                         \
        using type = TypeName;                                                      \
        static constexpr uint type_size = sizeof(type);                             \
                                                                                    \
    private:                                                                        \
        ByteBufferVar *_buffer{};                                                   \
        Uint _view_size{};                                                          \
        Uint _offset{};                                                             \
        uint _stride{};                                                             \
                                                                                    \
    public:                                                                         \
        SOAView() = default;                                                        \
        SOAView(ByteBufferVar &buffer, const Uint &view_size,                       \
                const Uint &ofs, uint stride)                                       \
            : _buffer(&buffer), _view_size(view_size),                              \
              _offset(ofs), _stride(stride) {}                                      \
                                                                                    \
        template<typename Index>                                                    \
        requires is_integral_expr_v<Index>                                          \
        [[nodiscard]] Var<float> read(Index &&index) const noexcept {               \
            return _buffer->load_as<type>(_offset + OC_FORWARD(index) * type_size); \
        }                                                                           \
                                                                                    \
        template<typename Index>                                                    \
        requires is_integral_expr_v<Index>                                          \
        void write(Index &&index, const Var<type> &val) noexcept {                  \
            _buffer->store(_offset + OC_FORWARD(index) * type_size, val);           \
        }                                                                           \
                                                                                    \
        template<typename int_type = uint>                                          \
        [[nodiscard]] Var<int_type> size_in_byte() const noexcept {                 \
            return _view_size / _stride * type_size;                                \
        }                                                                           \
    };

#define OC_COMMA ,

OC_MAKE_ATOMIC_SOA(uint, )
OC_MAKE_ATOMIC_SOA(uint64t, )
OC_MAKE_ATOMIC_SOA(float, )
OC_MAKE_ATOMIC_SOA(int, )
OC_MAKE_ATOMIC_SOA(array<T OC_COMMA N>,
                   typename T OC_COMMA uint N)

template<typename T>
struct SOAView<Vector<T, 2>> {
public:
    static constexpr uint type_size = sizeof(Vector<T, 2>);

public:
    SOAView<T> x;
    SOAView<T> y;

public:
    explicit SOAView(ByteBufferVar &buffer_var,
                     Uint view_size = InvalidUI32,
                     Uint offset = 0u,
                     uint stride = type_size) {
        view_size = ocarina::min(buffer_var.size<uint>(), view_size);
        x = SOAView<T>(buffer_var, view_size, offset, stride);
        offset += x.size_in_byte();
        y = SOAView<T>(buffer_var, view_size, offset, stride);
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<Vector<T, 2>> read(Index &&index) const noexcept {
        Var<Vector<T, 2>> ret;
        ret.x = x.read(OC_FORWARD(index));
        ret.y = y.read(OC_FORWARD(index));
        return ret;
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    void write(Index &&index, const Var<Vector<T, 2>> &val) noexcept {
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

template<typename T>
struct SOAView<Vector<T, 3>> {
public:
    static constexpr uint type_size = sizeof(Vector<T, 3>);

public:
    SOAView<T> x;
    SOAView<T> y;
    SOAView<T> z;

public:
    explicit SOAView(ByteBufferVar &buffer_var,
                     Uint view_size = InvalidUI32,
                     Uint offset = 0u,
                     uint stride = type_size) {
        view_size = ocarina::min(buffer_var.size<uint>(), view_size);
        x = SOAView<T>(buffer_var, view_size, offset, stride);
        offset += x.size_in_byte();
        y = SOAView<T>(buffer_var, view_size, offset, stride);
        offset += y.size_in_byte();
        z = SOAView<T>(buffer_var, view_size, offset, stride);
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<Vector<T, 3>> read(Index &&index) const noexcept {
        Var<Vector<T, 3>> ret;
        ret.x = x.read(OC_FORWARD(index));
        ret.y = y.read(OC_FORWARD(index));
        ret.z = z.read(OC_FORWARD(index));
        return ret;
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    void write(Index &&index, const Var<Vector<T, 3>> &val) noexcept {
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

template<typename T>
struct SOAView<Vector<T, 4>> {
public:
    static constexpr uint type_size = sizeof(Vector<T, 4>);

public:
    SOAView<T> x;
    SOAView<T> y;
    SOAView<T> z;
    SOAView<T> w;

public:
    explicit SOAView(ByteBufferVar &buffer_var,
                     Uint view_size = InvalidUI32,
                     Uint offset = 0u,
                     uint stride = type_size) {
        view_size = ocarina::min(buffer_var.size<uint>(), view_size);
        x = SOAView<T>(buffer_var, view_size, offset, stride);
        offset += x.size_in_byte();
        y = SOAView<T>(buffer_var, view_size, offset, stride);
        offset += y.size_in_byte();
        z = SOAView<T>(buffer_var, view_size, offset, stride);
        offset += z.size_in_byte();
        w = SOAView<T>(buffer_var, view_size, offset, stride);
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<Vector<T, 4>> read(Index &&index) const noexcept {
        Var<Vector<T, 4>> ret;
        ret.x = x.read(OC_FORWARD(index));
        ret.y = y.read(OC_FORWARD(index));
        ret.z = z.read(OC_FORWARD(index));
        ret.w = w.read(OC_FORWARD(index));
        return ret;
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    void write(Index &&index, const Var<Vector<T, 4>> &val) noexcept {
        x.write(OC_FORWARD(index), val.x);
        y.write(OC_FORWARD(index), val.y);
        z.write(OC_FORWARD(index), val.z);
        w.write(OC_FORWARD(index), val.w);
    }

    template<typename int_type = uint>
    [[nodiscard]] Var<int_type> size_in_byte() const noexcept {
        Var<int_type> ret = 0;
        ret += x.size_in_byte();
        ret += y.size_in_byte();
        ret += z.size_in_byte();
        ret += w.size_in_byte();
        return ret;
    }
};

template<uint N>
struct SOAView<Matrix<N>> {
public:
    static constexpr uint type_size = sizeof(Matrix<N>);

private:
    array<SOAView<Vector<float, N>>, N> _cols;

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
    [[nodiscard]] Var<Matrix<N>> read(Index &&index) const noexcept {
        Var<Matrix<N>> ret;
        for (int i = 0; i < N; ++i) {
            ret[i] = _cols[i].read(OC_FORWARD(index));
        }
        return ret;
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    void write(Index &&index, const Var<Matrix<N>> &val) noexcept {
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
