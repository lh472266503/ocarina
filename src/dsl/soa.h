//
// Created by Zero on 2024/2/25.
//

#pragma once

#include "core/basic_types.h"
#include "builtin.h"
#include "var.h"

namespace ocarina {

template<typename T>
struct BufferStorage {
public:
    using Storage = std::aligned_storage_t<sizeof(T), alignof(T)>;

private:
    Storage _storage{};

public:
    BufferStorage() = default;
    explicit BufferStorage(T &buffer) {
        oc_memcpy(addressof(_storage), addressof(buffer), sizeof(T));
    }

    BufferStorage(const BufferStorage<T> &other) {
        oc_memcpy(addressof(_storage), addressof(other._storage), sizeof(T));
    }

    BufferStorage &operator=(const BufferStorage<T> &other) {
        oc_memcpy(addressof(_storage), addressof(other._storage), sizeof(T));
        return *this;
    }

    [[nodiscard]] T *operator->() noexcept {
        return reinterpret_cast<T *>(addressof(_storage));
    }

    [[nodiscard]] const T *operator->() const noexcept {
        return reinterpret_cast<const T *>(addressof(_storage));
    }
};

template<typename T, typename TBuffer>
struct SOAView {
    static_assert(always_false_v<T, TBuffer>, "The SOAView template must be specialized");
};

#define OC_MAKE_ATOMIC_SOA(TemplateArgs, TypeName)                                          \
    TemplateArgs struct SOAView<TypeName, TBuffer> {                                        \
    public:                                                                                 \
        using element_type = TypeName;                                                      \
        static constexpr uint type_size = sizeof(element_type);                             \
                                                                                            \
    private:                                                                                \
        BufferStorage<TBuffer> _buffer{};                                                   \
        Uint _view_size{};                                                                  \
        Uint _offset{};                                                                     \
        uint _stride{};                                                                     \
                                                                                            \
    public:                                                                                 \
        SOAView() = default;                                                                \
        SOAView(TBuffer &buffer, const Uint &view_size = InvalidUI32,                       \
                const Uint &ofs = 0u, uint stride = type_size)                              \
            : _buffer(buffer),                                                              \
              _view_size(ocarina::min(view_size, buffer.size_in_byte<uint>())),             \
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

OC_MAKE_ATOMIC_SOA(template<typename TBuffer>, uint)
OC_MAKE_ATOMIC_SOA(template<typename TBuffer>, uint64t)
OC_MAKE_ATOMIC_SOA(template<typename TBuffer>, float)
OC_MAKE_ATOMIC_SOA(template<typename TBuffer>, int)
OC_MAKE_ATOMIC_SOA(template<typename T OC_COMMA uint N OC_COMMA typename TBuffer>,
                   Vector<T OC_COMMA N>)

}// namespace ocarina

#define OC_MAKE_SOA_MEMBER(field_name) ocarina::SOAView<decltype(element_type::field_name), TBuffer> field_name;

#define OC_MAKE_SOA_MEMBER_CONSTRUCT(field_name)                                                                       \
    field_name = ocarina::SOAView<decltype(element_type::field_name), TBuffer>(buffer_var, view_size, offset, stride); \
    offset += field_name.size_in_byte();

#define OC_MAKE_SOA_MEMBER_READ(field_name) ret.field_name = field_name.read(OC_FORWARD(index));

#define OC_MAKE_SOA_MEMBER_WRITE(field_name) field_name.write(OC_FORWARD(index), val.field_name);

#define OC_MAKE_SOA_MEMBER_SIZE(field_name) ret += field_name.size_in_byte();

#define OC_MAKE_STRUCT_SOA_VIEW(TemplateArgs, S, ...)                             \
    TemplateArgs struct ocarina::SOAView<S, TBuffer> {                            \
    public:                                                                       \
        using element_type = S;                                                   \
        static constexpr uint type_size = sizeof(element_type);                   \
                                                                                  \
    public:                                                                       \
        MAP(OC_MAKE_SOA_MEMBER, ##__VA_ARGS__)                                    \
    public:                                                                       \
        SOAView() = default;                                                      \
        explicit SOAView(TBuffer &buffer_var,                                     \
                         Uint view_size = InvalidUI32,                            \
                         Uint offset = 0u,                                        \
                         uint stride = type_size) {                               \
            view_size = ocarina::min(buffer_var.size_in_byte<uint>(), view_size); \
            MAP(OC_MAKE_SOA_MEMBER_CONSTRUCT, ##__VA_ARGS__)                      \
        }                                                                         \
                                                                                  \
        template<typename Index>                                                  \
        requires is_integral_expr_v<Index>                                        \
        [[nodiscard]] Var<element_type> read(Index &&index) const noexcept {      \
            Var<element_type> ret;                                                \
            MAP(OC_MAKE_SOA_MEMBER_READ, ##__VA_ARGS__)                           \
            return ret;                                                           \
        }                                                                         \
                                                                                  \
        template<typename Index>                                                  \
        requires is_integral_expr_v<Index>                                        \
        void write(Index &&index, const Var<element_type> &val) noexcept {        \
            MAP(OC_MAKE_SOA_MEMBER_WRITE, ##__VA_ARGS__)                          \
        }                                                                         \
        template<typename int_type = uint>                                        \
        [[nodiscard]] Var<int_type> size_in_byte() const noexcept {               \
            Var<int_type> ret = 0;                                                \
            MAP(OC_MAKE_SOA_MEMBER_SIZE, ##__VA_ARGS__)                           \
            return ret;                                                           \
        }                                                                         \
    };

OC_MAKE_STRUCT_SOA_VIEW(template<typename T OC_COMMA typename TBuffer>, ocarina::Vector<T OC_COMMA 2>, x, y)
OC_MAKE_STRUCT_SOA_VIEW(template<typename T OC_COMMA typename TBuffer>, ocarina::Vector<T OC_COMMA 3>, x, y, z)
OC_MAKE_STRUCT_SOA_VIEW(template<typename T OC_COMMA typename TBuffer>, ocarina::Vector<T OC_COMMA 4>, x, y, z, w)

namespace ocarina {

template<uint N, typename TBuffer>
struct SOAView<Matrix<N>, TBuffer> {
public:
    using element_type = Matrix<N>;
    using array_element_type = decltype(std::declval<element_type>()[0]);
    static constexpr uint type_size = sizeof(element_type);

private:
    array<SOAView<Vector<float, N>, TBuffer>, N> _array{};

public:
    SOAView() = default;
    explicit SOAView(TBuffer &buffer_var,
                     Uint view_size = InvalidUI32,
                     Uint offset = 0u,
                     uint stride = type_size) {
        view_size = ocarina::min(buffer_var.size_in_byte<uint>(), view_size);
        for (int i = 0; i < N; ++i) {
            _array[i] = SOAView<Vector<float, N>, TBuffer>(buffer_var, view_size,
                                                          offset, stride);
            offset += _array[i].size_in_byte();
        }
    }

    [[nodiscard]] auto &operator[](size_t index) const noexcept { return _array[index]; }
    [[nodiscard]] auto &operator[](size_t index) noexcept { return _array[index]; }

    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<element_type> read(Index &&index) const noexcept {
        Var<element_type> ret;
        for (int i = 0; i < N; ++i) {
            ret[i] = _array[i].read(OC_FORWARD(index));
        }
        return ret;
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    void write(Index &&index, const Var<element_type> &val) noexcept {
        for (int i = 0; i < N; ++i) {
            _array[i].write(OC_FORWARD(index), val[i]);
        }
    }

    template<typename int_type = uint>
    [[nodiscard]] Var<int_type> size_in_byte() const noexcept {
        Var<int_type> ret = 0;
        for (int i = 0; i < N; ++i) {
            ret += _array[i].size_in_byte();
        }
        return ret;
    }
};

#undef OC_MAKE_ATOMIC_SOA

template<typename Elm, typename TBuffer>
[[nodiscard]] SOAView<Elm, TBuffer> make_soa_view(TBuffer &buffer) noexcept {
    return SOAView<Elm, TBuffer>(buffer);
}

}// namespace ocarina
