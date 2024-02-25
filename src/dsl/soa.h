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

#define OC_MAKE_SOA_ATOMIC_BODY(TypeName)                                          \
    struct SOAView<TypeName> {                                                     \
    public:                                                                        \
        static constexpr uint size = sizeof(TypeName);                             \
        using type = TypeName;                                                     \
                                                                                   \
    private:                                                                       \
        Uint _offset{};                                                            \
        const ByteBufferVar *_buffer{};                                            \
        uint _stride{};                                                            \
                                                                                   \
    public:                                                                        \
        SOAView(const Uint &ofs, const ByteBufferVar &buffer, uint stride)         \
            : _offset(ofs), _buffer(&buffer), _stride(stride) {}                   \
                                                                                   \
        template<typename Index>                                                   \
        [[nodiscard]] Var<float> read(Index &&index) noexcept {                    \
            return _buffer->load_as<TypeName>(_offset + OC_FORWARD(index) * size); \
        }                                                                          \
                                                                                   \
        template<typename int_type = uint>                                         \
        [[nodiscard]] Var<int_type> size_in_byte() noexcept {                      \
            return _buffer->size<int_type>() / _stride * size;                     \
        }                                                                          \
    };

#define OC_MAKE_SCALAR_SOA_ATOMIC(TypeName) \
    template<>                              \
    OC_MAKE_SOA_ATOMIC_BODY(TypeName)

OC_MAKE_SCALAR_SOA_ATOMIC(uint)
OC_MAKE_SCALAR_SOA_ATOMIC(uint64t)
OC_MAKE_SCALAR_SOA_ATOMIC(float)
OC_MAKE_SCALAR_SOA_ATOMIC(int)

#define OC_COMMA ,

template<typename T, uint N>
OC_MAKE_SOA_ATOMIC_BODY(array<T OC_COMMA N>)

#undef OC_MAKE_SCALAR_SOA_ATOMIC
#undef OC_MAKE_SOA_ATOMIC_BODY

}// namespace ocarina
