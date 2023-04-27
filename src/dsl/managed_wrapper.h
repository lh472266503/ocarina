//
// Created by zhu on 2023/4/27.
//

#pragma once

#include "serialize.h"
#include "rhi/resources/managed.h"

namespace ocarina {

template<typename T>
class ManagedWrapper : public Managed<T>,
                       public ISerializable<ScalarUnion> {
public:
    using Super = Managed<T>;

private:
    ResourceArray *_resource_array{};
    Serialize<uint> _index{InvalidUI32};
    Serialize<uint> _length{InvalidUI32};

public:
    ManagedWrapper() = default;
    OC_SERIALIZABLE_FUNC(ScalarUnion, _index, _length)
    void init(ResourceArray &resource_array) noexcept { _resource_array = &resource_array; }

    explicit ManagedWrapper(ResourceArray &resource_array) : _resource_array(&resource_array) {}
    void register_self() noexcept {
        _index = _resource_array->emplace(Super::device());
        _length = [&]() {
            return static_cast<uint>(Super::host().size());
        };
    }

    [[nodiscard]] const Serialize<uint> &index() const noexcept { return _index; }
    [[nodiscard]] const Serialize<uint> &length() const noexcept { return _length; }

    template<typename Index>
    requires concepts::all_integral<expr_value_t<Index>>
    OC_NODISCARD auto read(Index &&index) const noexcept {
        return _resource_array->buffer<T>(_index.hv()).read(OC_FORWARD(index));
    }

    template<typename Target, typename Offset>
    requires is_integral_expr_v<Offset>
    OC_NODISCARD auto byte_read(Offset &&offset) const noexcept {
        return _resource_array->byte_buffer(_index.hv()).read<Target>(OC_FORWARD(offset));
    }

    template<typename Elm, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] Array<Elm> read_dynamic_array(uint size, Offset &&offset) const noexcept {
        return _resource_array->byte_buffer(_index.hv()).read_dynamic_array<Elm>(size, OC_FORWARD(offset));
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<T, expr_value_t<Val>>
    void write(Index &&index, Val &&elm) {
        _resource_array->buffer<T>(_index.hv()).write(OC_FORWARD(index), OC_FORWARD(elm));
    }
};

}// namespace ocarina