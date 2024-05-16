//
// Created by Zero on 2024/2/25.
//

#pragma once

#include "math/basic_types.h"
#include "builtin.h"
#include "var.h"
#include "type_trait.h"
#include "func.h"

namespace ocarina {

template<typename T>
struct BufferStorage {
public:
    using Storage = std::aligned_storage_t<sizeof(T), alignof(T)>;

private:
    Storage _storage{};

public:
    BufferStorage() = default;
    explicit BufferStorage(const T &buffer) {
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

#define OC_MAKE_ATOMIC_SOA(TemplateArgs, TypeName)                                        \
    TemplateArgs struct ocarina::SOAView<TypeName, TBuffer> {                             \
    public:                                                                               \
        static_assert(is_valid_buffer_element_v<TypeName>);                               \
        using atomic_type = TypeName;                                                     \
        static constexpr ocarina::uint type_size = sizeof(atomic_type);                   \
                                                                                          \
    private:                                                                              \
        ocarina::BufferStorage<TBuffer> buffer_{};                                        \
        ocarina::Uint _view_size{};                                                       \
        ocarina::Uint _offset{};                                                          \
        ocarina::uint _stride{};                                                          \
                                                                                          \
    public:                                                                               \
        SOAView() = default;                                                              \
        explicit SOAView(const TBuffer &buffer,                                           \
                         const ocarina::Uint &view_size = ocarina::InvalidUI32,           \
                         const ocarina::Uint &ofs = 0u, ocarina::uint stride = type_size) \
            : buffer_(buffer),                                                            \
              _view_size(ocarina::min(view_size,                                          \
                                      buffer.template size_in_byte<ocarina::uint>())),    \
              _offset(ofs), _stride(stride) {}                                            \
                                                                                          \
        template<typename Index>                                                          \
        requires ocarina::is_integral_expr_v<Index>                                       \
        [[nodiscard]] ocarina::Var<atomic_type> read(Index &&index) const noexcept {      \
            return buffer_->template load_as<atomic_type>(_offset +                       \
                                                          OC_FORWARD(index) * type_size); \
        }                                                                                 \
                                                                                          \
        template<typename Index>                                                          \
        requires ocarina::is_integral_expr_v<Index>                                       \
        void write(Index &&index, const ocarina::Var<atomic_type> &val) noexcept {        \
            buffer_->store(_offset + OC_FORWARD(index) * type_size, val);                 \
        }                                                                                 \
                                                                                          \
        template<typename int_type = ocarina::uint>                                       \
        [[nodiscard]] ocarina::Var<int_type> size_in_byte() const noexcept {              \
            return _view_size / _stride * type_size;                                      \
        }                                                                                 \
    };

}// namespace ocarina

OC_MAKE_ATOMIC_SOA(template<typename TBuffer>, ocarina::uint)
OC_MAKE_ATOMIC_SOA(template<typename TBuffer>, ocarina::uint64t)
OC_MAKE_ATOMIC_SOA(template<typename TBuffer>, float)
OC_MAKE_ATOMIC_SOA(template<typename TBuffer>, int)
OC_MAKE_ATOMIC_SOA(template<typename T OC_COMMA ocarina::uint N OC_COMMA typename TBuffer>,
                   ocarina::Vector<T OC_COMMA N>)

#define OC_MAKE_SOA_MEMBER(field_name) ocarina::SOAView<decltype(struct_type::field_name), TBuffer> field_name;

#define OC_MAKE_SOA_MEMBER_CONSTRUCT(field_name)                                                                      \
    field_name = ocarina::SOAView<decltype(struct_type::field_name), TBuffer>(buffer_var, view_size, offset, stride); \
    offset += field_name.size_in_byte();

#define OC_MAKE_SOA_MEMBER_READ(field_name) ret.field_name = field_name.read(OC_FORWARD(index));

#define OC_MAKE_SOA_MEMBER_WRITE(field_name) field_name.write(OC_FORWARD(index), val.field_name);

#define OC_MAKE_SOA_MEMBER_SIZE(field_name) ret += field_name.size_in_byte();

#define OC_MAKE_STRUCT_SOA_VIEW(TemplateArgs, S, ...)                                   \
    TemplateArgs struct ocarina::SOAView<S, TBuffer> {                                  \
    public:                                                                             \
        using struct_type = S;                                                          \
        static_assert(is_valid_buffer_element_v<struct_type>);                          \
        static constexpr ocarina::uint type_size = sizeof(struct_type);                 \
                                                                                        \
    public:                                                                             \
        MAP(OC_MAKE_SOA_MEMBER, ##__VA_ARGS__)                                          \
    public:                                                                             \
        SOAView() = default;                                                            \
        explicit SOAView(TBuffer &buffer_var,                                           \
                         ocarina::Uint view_size = InvalidUI32,                         \
                         ocarina::Uint offset = 0u,                                     \
                         ocarina::uint stride = type_size) {                            \
            view_size = ocarina::min(buffer_var.template size_in_byte<ocarina::uint>(), \
                                     view_size);                                        \
            MAP(OC_MAKE_SOA_MEMBER_CONSTRUCT, ##__VA_ARGS__)                            \
        }                                                                               \
                                                                                        \
        template<typename Index>                                                        \
        requires ocarina::is_integral_expr_v<Index>                                     \
        [[nodiscard]] ocarina::Var<struct_type> read(Index &&index) const noexcept {    \
            return ocarina::outline(                                                    \
                [&] {                                                                   \
                    ocarina::Var<struct_type> ret;                                      \
                    MAP(OC_MAKE_SOA_MEMBER_READ, ##__VA_ARGS__)                         \
                    return ret;                                                         \
                },                                                                      \
                "SOAView<" #S ">::read");                                               \
        }                                                                               \
                                                                                        \
        template<typename Index>                                                        \
        requires ocarina::is_integral_expr_v<Index>                                     \
        void write(Index &&index, const ocarina::Var<struct_type> &val) noexcept {      \
            ocarina::outline(                                                           \
                [&] {                                                                   \
                    MAP(OC_MAKE_SOA_MEMBER_WRITE, ##__VA_ARGS__)                        \
                },                                                                      \
                "SOAView<" #S ">::write");                                              \
        }                                                                               \
        template<typename int_type = ocarina::uint>                                     \
        [[nodiscard]] ocarina::Var<int_type> size_in_byte() const noexcept {            \
            return ocarina::outline(                                                    \
                [&] {                                                                   \
                    ocarina::Var<int_type> ret = 0;                                     \
                    MAP(OC_MAKE_SOA_MEMBER_SIZE, ##__VA_ARGS__)                         \
                    return ret;                                                         \
                },                                                                      \
                "SOAView<" #S ">::size_in_byte");                                       \
        }                                                                               \
    };

//OC_MAKE_STRUCT_SOA_VIEW(template<typename T OC_COMMA typename TBuffer>, ocarina::Vector<T OC_COMMA 2>, x, y)
//OC_MAKE_STRUCT_SOA_VIEW(template<typename T OC_COMMA typename TBuffer>, ocarina::Vector<T OC_COMMA 3>, x, y, z)
//OC_MAKE_STRUCT_SOA_VIEW(template<typename T OC_COMMA typename TBuffer>, ocarina::Vector<T OC_COMMA 4>, x, y, z, w)

#define OC_MAKE_ARRAY_SOA_VIEW(TemplateArgs, TypeName, ElementType)                     \
    TemplateArgs struct ocarina::SOAView<TypeName, TBuffer> {                           \
    public:                                                                             \
        using struct_type = TypeName;                                                   \
        static_assert(is_valid_buffer_element_v<struct_type>);                          \
        using element_type = ElementType;                                               \
        static constexpr ocarina::uint type_size = sizeof(struct_type);                 \
                                                                                        \
    private:                                                                            \
        ocarina::array<ocarina::SOAView<element_type, TBuffer>, N> _array{};            \
                                                                                        \
    public:                                                                             \
        SOAView() = default;                                                            \
        explicit SOAView(TBuffer &buffer_var,                                           \
                         ocarina::Uint view_size = InvalidUI32,                         \
                         ocarina::Uint offset = 0u,                                     \
                         ocarina::uint stride = type_size) {                            \
            view_size = ocarina::min(buffer_var.template size_in_byte<ocarina::uint>(), \
                                     view_size);                                        \
            for (int i = 0; i < N; ++i) {                                               \
                _array[i] = SOAView<element_type, TBuffer>(buffer_var, view_size,       \
                                                           offset, stride);             \
                offset += _array[i].template size_in_byte<ocarina::uint>();             \
            }                                                                           \
        }                                                                               \
                                                                                        \
        [[nodiscard]] auto &operator[](size_t index) const noexcept {                   \
            return _array[index];                                                       \
        }                                                                               \
        [[nodiscard]] auto &operator[](size_t index) noexcept {                         \
            return _array[index];                                                       \
        }                                                                               \
                                                                                        \
        template<typename Index>                                                        \
        requires ocarina::is_integral_expr_v<Index>                                     \
        [[nodiscard]] ocarina::Var<struct_type> read(Index &&index) const noexcept {    \
            return ocarina::outline(                                                    \
                [&] {                                                                   \
                    ocarina::Var<struct_type> ret;                                      \
                    for (int i = 0; i < N; ++i) {                                       \
                        ret[i] = _array[i].read(OC_FORWARD(index));                     \
                    }                                                                   \
                    return ret;                                                         \
                },                                                                      \
                "SOAView<" #TypeName ">::read");                                        \
        }                                                                               \
                                                                                        \
        template<typename Index>                                                        \
        requires ocarina::is_integral_expr_v<Index>                                     \
        void write(Index &&index, const ocarina::Var<struct_type> &val) noexcept {      \
            ocarina::outline(                                                           \
                [&] {                                                                   \
                    for (int i = 0; i < N; ++i) {                                       \
                        _array[i].write(OC_FORWARD(index), val[i]);                     \
                    }                                                                   \
                },                                                                      \
                "SOAView<" #TypeName ">::write");                                       \
        }                                                                               \
                                                                                        \
        template<typename int_type = ocarina::uint>                                     \
        [[nodiscard]] ocarina::Var<int_type> size_in_byte() const noexcept {            \
            return ouline(                                                              \
                [&] {                                                                   \
                    ocarina::Var<int_type> ret = 0;                                     \
                    for (int i = 0; i < N; ++i) {                                       \
                        ret += _array[i].size_in_byte();                                \
                    }                                                                   \
                    return ret;                                                         \
                },                                                                      \
                "SOAView<" #TypeName ">::size_in_byte");                                \
        }                                                                               \
    };

OC_MAKE_ARRAY_SOA_VIEW(template<ocarina::uint N OC_COMMA typename TBuffer>,
                       ocarina::Matrix<N>, Vector<float OC_COMMA N>)
OC_MAKE_ARRAY_SOA_VIEW(template<ocarina::uint N OC_COMMA typename T OC_COMMA typename TBuffer>,
                       ocarina::array<T OC_COMMA N>, T)

namespace ocarina {
template<typename Elm, typename TBuffer>
[[nodiscard]] SOAView<Elm, TBuffer> make_soa_view(TBuffer &buffer) noexcept {
    return SOAView<Elm, TBuffer>(buffer);
}
}// namespace ocarina
