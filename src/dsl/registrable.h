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
        return _bindless_array->buffer_var<T>(buffer_index).read(access_index);
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

    void register_self(size_t offset = 0, size_t size = 0) noexcept {
        BufferView<T> buffer_view = super().view(offset, size);
        _index = _bindless_array->emplace(buffer_view);
        _length = [=]() {
            return static_cast<uint>(buffer_view.size());
        };
    }

    uint register_view(size_t offset, size_t size = 0) {
        BufferView<T> buffer_view = super().view(offset, size);
        return _bindless_array->emplace(buffer_view);
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
            _bindless_array->buffer_var<T>(*_index).write(OC_FORWARD(index), OC_FORWARD(elm));
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
        return _bindless_array->byte_buffer_var(*_index).template read<Target>(OC_FORWARD(offset));
    }

    template<typename Elm, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] DynamicArray<Elm> load_dynamic_array(uint size, Offset &&offset) const noexcept {
        OC_ASSERT(has_registered());
        return _bindless_array->byte_buffer_var(*_index).template load_dynamic_array<Elm>(size, OC_FORWARD(offset));
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<T, expr_value_t<Val>>
    void write(Index &&index, Val &&elm) {
        if (!has_registered()) {
            Super::write(OC_FORWARD(index), OC_FORWARD(elm));
        } else {
            _bindless_array->buffer_var<T>(*_index).write(OC_FORWARD(index), OC_FORWARD(elm));
        }
    }
};

class RegistrableTexture : public ManagedTexture, public Registrable {
public:
    RegistrableTexture() = default;
    explicit RegistrableTexture(BindlessArray &bindless_array) : Registrable(&bindless_array) {}
    void register_self() noexcept {
        if (has_registered()) {
            _bindless_array->set_texture(_index.hv(), *this);
        } else {
            _index = _bindless_array->emplace(*this);
        }
        _length = 0;
    }

    void unregister() noexcept {
        if (has_registered()) {
            (*_bindless_array)->remove_texture(_index.hv());
            _index = InvalidUI32;
        }
    }

    template<typename ...Args>
    OC_NODISCARD auto sample(uint channel_num, Args &&...args) const noexcept {
        if (has_registered()) {
            return _bindless_array->tex_var(*_index).sample(channel_num, OC_FORWARD(args)...);
        } else {
            return Texture::sample(channel_num, OC_FORWARD(args)...);
        }
    }

    template<typename Target, typename ...Args>
    OC_NODISCARD auto read(Args &&...args) const noexcept {
        return Texture::read<Target>(OC_FORWARD(args)...);
    }

    template<typename ...Args>
    OC_NODISCARD auto write(Args &&...args) noexcept {
        return Texture::write(OC_FORWARD(args)...);
    }
};

}// namespace ocarina