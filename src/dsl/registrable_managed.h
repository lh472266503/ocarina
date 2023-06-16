//
// Created by zhu on 2023/4/27.
//

#pragma once

#include "serialize.h"
#include "rhi/resources/managed.h"

namespace ocarina {

template<typename T>
class RegistrableManaged : public Managed<T>,
                           public Serializable<serialize_element_ty> {
public:
    using Super = Managed<T>;

private:
    ResourceArray *_resource_array{};
    Serial<uint> _index{InvalidUI32};
    Serial<uint> _length{InvalidUI32};

public:
    RegistrableManaged() = default;
    OC_SERIALIZABLE_FUNC(Serializable<serialize_element_ty>, _index, _length)
    void init(ResourceArray &resource_array) noexcept { _resource_array = &resource_array; }

    explicit RegistrableManaged(ResourceArray &resource_array) : _resource_array(&resource_array) {}
    void register_self() noexcept {
        _index = _resource_array->emplace(Super::device_buffer());
        _length = [&]() {
            return static_cast<uint>(Super::host_buffer().size());
        };
    }

    [[nodiscard]] bool has_registered() const noexcept { return _index.hv() != InvalidUI32; }
    [[nodiscard]] const Serial<uint> &index() const noexcept { return _index; }
    [[nodiscard]] const Serial<uint> &length() const noexcept { return _length; }

    template<typename Index>
    requires concepts::all_integral<expr_value_t<Index>>
    OC_NODISCARD auto read(Index &&index) const noexcept {
        if (!has_registered()) {
            return Super::read(OC_FORWARD(index));
        }
        return _resource_array->buffer<T>(*_index).read(OC_FORWARD(index));
    }

    template<typename Target, typename Offset>
    requires is_integral_expr_v<Offset>
    OC_NODISCARD auto byte_read(Offset &&offset) const noexcept {
        OC_ASSERT(has_registered());
        return _resource_array->byte_buffer(*_index).read<Target>(OC_FORWARD(offset));
    }

    template<typename Elm, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] Array<Elm> read_dynamic_array(uint size, Offset &&offset) const noexcept {
        OC_ASSERT(has_registered());
        return _resource_array->byte_buffer(*_index).read_dynamic_array<Elm>(size, OC_FORWARD(offset));
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<T, expr_value_t<Val>>
    void write(Index &&index, Val &&elm) {
        if (!has_registered()) {
            Super::write(OC_FORWARD(index), OC_FORWARD(elm));
        } else {
            _resource_array->buffer<T>(*_index).write(OC_FORWARD(index), OC_FORWARD(elm));
        }
    }
};

}// namespace ocarina