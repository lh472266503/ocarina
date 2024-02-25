//
// Created by Zero on 2024/2/25.
//

#pragma once

#include "core/basic_types.h"
#include "var.h"

namespace ocarina {

template<typename T>
struct SOAView {
    static_assert(always_false_v<T>);
};

#define OC_MAKE_ATOMIC_SOA(TypeName, TemplateArg)                              \
    template<TemplateArg>                                                      \
    struct SOAView<TypeName> {                                                 \
    public:                                                                    \
        using type = TypeName;                                                 \
        static constexpr uint size = sizeof(type);                             \
                                                                               \
    private:                                                                   \
        Uint _offset{};                                                        \
        const ByteBufferVar *_buffer{};                                        \
        uint _stride{};                                                        \
                                                                               \
    public:                                                                    \
        SOAView(const Uint &ofs, const ByteBufferVar &buffer, uint stride)     \
            : _offset(ofs), _buffer(&buffer), _stride(stride) {}               \
                                                                               \
        template<typename Index>                                               \
        [[nodiscard]] Var<float> read(Index &&index) noexcept {                \
            return _buffer->load_as<type>(_offset + OC_FORWARD(index) * size); \
        }                                                                      \
                                                                               \
        template<typename int_type = uint>                                     \
        [[nodiscard]] Var<int_type> size_in_byte() noexcept {                  \
            return _buffer->size<int_type>() / _stride * size;                 \
        }                                                                      \
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
    SOAView<T> x;
    SOAView<T> y;
};

template<typename T>
struct SOAView<Vector<T, 3>> {
    SOAView<T> x;
    SOAView<T> y;
    SOAView<T> z;
};

template<typename T>
struct SOAView<Vector<T, 4>> {
    SOAView<T> x;
    SOAView<T> y;
    SOAView<T> z;
    SOAView<T> w;
};

template<>
struct SOAView<Matrix<2>> {
    array<SOAView<float2>, 2> cols;

};

template<>
struct SOAView<Matrix<3>> {
    array<SOAView<float3>, 3> cols;

};

template<>
struct SOAView<Matrix<4>> {
    array<SOAView<float4>, 4> cols;
    
};

#undef OC_MAKE_ATOMIC_SOA

}// namespace ocarina
