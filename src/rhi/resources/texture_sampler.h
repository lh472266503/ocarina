//
// Created by Zero on 2023/12/29.
//

#pragma once

#include "core/basic_types.h"

namespace ocarina {

class TextureSampler {
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
        ZERO
    };
private:
    Filter _u_filter{Filter::POINT};
    Filter _v_filter{Filter::POINT};
    Filter _w_filter{Filter::POINT};
    Address _u_address{Address::EDGE};
    Address _v_address{Address::EDGE};
    Address _w_address{Address::EDGE};

public:
    constexpr TextureSampler() noexcept = default;
};

}