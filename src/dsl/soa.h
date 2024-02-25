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

template<>
struct SOAView<uint> {
private:
    Uint _offset{};
    const ByteBufferVar *_buffer{};
    uint _stride{};

public:
    SOAView(const Uint &ofs, const ByteBufferVar &buffer, uint stride)
        : _offset(ofs), _buffer(&buffer), _stride(stride) {}

    template<typename Index>
    [[nodiscard]] Var<uint> read(Index &&index) noexcept {
        return _buffer->load_as<uint>(_offset + OC_FORWARD(index) * sizeof(uint));
    }

    template<typename int_type = uint>
    [[nodiscard]] Var<int_type> size_in_byte() noexcept {
        return _buffer->size<int_type>() / _stride * uint(sizeof(uint));
    }
};

}// namespace ocarina
