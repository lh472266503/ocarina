//
// Created by zhu on 2023/4/27.
//

#pragma once

#include "serialize.h"
#include "rhi/resources/managed.h"
#include "dsl/printer.h"
#include "core/platform.h"

namespace ocarina {

class Registrable : public Serializable<serialize_element_ty> {
protected:
    Serial<uint> _index{InvalidUI32};
    Serial<uint> _length{InvalidUI32};
    BindlessArray *_bindless_array{};

public:
    Registrable() = default;
    explicit Registrable(BindlessArray *bindless_array)
        : _bindless_array(bindless_array) {}
    OC_SERIALIZABLE_FUNC(Serializable<serialize_element_ty>, _index, _length)
    void set_bindless_array(BindlessArray &bindless_array) noexcept {
        _bindless_array = &bindless_array;
    }
    [[nodiscard]] BindlessArray *bindless_array() const noexcept {
        return _bindless_array;
    }
    [[nodiscard]] bool has_registered() const noexcept { return _index.hv() != InvalidUI32; }
    [[nodiscard]] const Serial<uint> &index() const noexcept { return _index; }
    [[nodiscard]] const Serial<uint> &length() const noexcept { return _length; }

protected:
    template<typename T, typename Index>
    requires concepts::all_integral<expr_value_t<Index>>
    OC_NODISCARD auto _read(Index &&index) const noexcept {
        Uint buffer_index = *_index;
        Uint access_index = OC_FORWARD(index);
        return _bindless_array->buffer<T>(buffer_index).read(access_index);
    }
};

template<typename T>
class RegistrableBuffer : public Buffer<T>,
                          public Registrable {
public:
    using Super = Buffer<T>;

public:
    explicit RegistrableBuffer(BindlessArray &bindless_array)
        : Super(), Registrable(&bindless_array) {}

    RegistrableBuffer() = default;
    void register_self() noexcept {
        _index = _bindless_array->emplace(super());
        _length = [&]() {
            return static_cast<uint>(Super::size());
        };
    }

    [[nodiscard]] Super &super() noexcept { return *this; }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    OC_NODISCARD auto read(Index &&index) const noexcept {
        if (!has_registered()) {
            return Super::read(OC_FORWARD(index));
        }
        return _read<T>(OC_FORWARD(index));
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<T, expr_value_t<Val>>
    void write(Index &&index, Val &&elm) {
        if (!has_registered()) {
            Super::write(OC_FORWARD(index), OC_FORWARD(elm));
        } else {
            _bindless_array->buffer<T>(*_index).write(OC_FORWARD(index), OC_FORWARD(elm));
        }
    }
};

template<typename T>
class RegistrableManaged : public Managed<T>,
                           public Registrable {
public:
    using Super = Managed<T>;

public:
    RegistrableManaged() = default;
    explicit RegistrableManaged(BindlessArray &bindless_array) : Registrable(&bindless_array) {}
    void register_self() noexcept {
        if (has_registered()) {
            _bindless_array->set_buffer(_index.hv(), Super::device_buffer());
        } else {
            _index = _bindless_array->emplace(Super::device_buffer());
        }
        _length = [&]() {
            return static_cast<uint>(Super::device_buffer().size());
        };
    }

    void unregister() noexcept {
        if (has_registered()) {
            (*_bindless_array)->remove_buffer(_index.hv());
            _index = InvalidUI32;
        }
    }

    template<typename Index>
    requires concepts::all_integral<expr_value_t<Index>>
    OC_NODISCARD auto read(Index &&index) const noexcept {
        if (!has_registered()) {
            return Super::read(OC_FORWARD(index));
        }
        return _read<T>(OC_FORWARD(index));
    }

    template<typename Target, typename Offset>
    requires is_integral_expr_v<Offset>
    OC_NODISCARD auto byte_read(Offset &&offset) const noexcept {
        OC_ASSERT(has_registered());
        return _bindless_array->byte_buffer(*_index).template read<Target>(OC_FORWARD(offset));
    }

    template<typename Elm, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] Array<Elm> read_dynamic_array(uint size, Offset &&offset) const noexcept {
        OC_ASSERT(has_registered());
        return _bindless_array->byte_buffer(*_index).template read_dynamic_array<Elm>(size, OC_FORWARD(offset));
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<T, expr_value_t<Val>>
    void write(Index &&index, Val &&elm) {
        if (!has_registered()) {
            Super::write(OC_FORWARD(index), OC_FORWARD(elm));
        } else {
            _bindless_array->buffer<T>(*_index).write(OC_FORWARD(index), OC_FORWARD(elm));
        }
    }
};

}// namespace ocarina