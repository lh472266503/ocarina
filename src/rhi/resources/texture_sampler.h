//
// Created by Zero on 2023/12/29.
//

#pragma once

#include "core/basic_types.h"
#include "core/hash.h"

namespace ocarina {

class TextureSampler : Hashable {
public:
    enum struct Filter : uint8_t {
        POINT,
        LINEAR_POINT,
        LINEAR_LINEAR,
        ANISOTROPIC
    };

    enum struct Address : uint8_t {
        EDGE,
        REPEAT,
        MIRROR,
        CLAMP
    };

private:
    Filter _filter{Filter::POINT};
    Address _u_address{Address::EDGE};
    Address _v_address{Address::EDGE};
    Address _w_address{Address::EDGE};

protected:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override {
        return hash64(_filter, _u_address, _v_address, _w_address);
    }

public:
    constexpr TextureSampler() noexcept = default;
    constexpr TextureSampler(Filter filter, Address address) noexcept
        : _filter{filter},
          _u_address{address},
          _v_address{address},
          _w_address{address} {}
    constexpr TextureSampler(Filter filter, Address u_address, Address v_address, Address w_address) noexcept
        : _filter{filter},
          _u_address{u_address},
          _v_address{v_address},
          _w_address{w_address} {}
    constexpr TextureSampler(Filter filter, Address u_address, Address v_address) noexcept
        : _filter{filter},
          _u_address{u_address},
          _v_address{v_address},
          _w_address{v_address} {}

    OC_MAKE_MEMBER_GETTER_SETTER(filter,)
    OC_MAKE_MEMBER_GETTER_SETTER(u_address,)
    OC_MAKE_MEMBER_GETTER_SETTER(v_address,)
    OC_MAKE_MEMBER_GETTER_SETTER(w_address,)

    [[nodiscard]] bool operator==(TextureSampler rhs) const noexcept {
        return hash() == rhs.hash();
    }
};

}// namespace ocarina