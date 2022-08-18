//
// Created by Zero on 27/07/2022.
//

#pragma once

#include "core/basic_traits.h"
#include "dsl/operators.h"
#include "dsl/struct.h"

namespace ocarina {

struct alignas(16) Hit {
    uint inst_id{uint(-1)};
    uint prim_id{uint(-1)};
    float2 bary;
};

}// namespace ocarina

OC_STRUCT(ocarina::Hit, inst_id, prim_id, bary){
    void init(){
        inst_id = uint(-1);
    }

    [[nodiscard]] auto is_miss() noexcept {
        return make_expr(inst_id == uint(-1));
    }
};

constexpr float ray_t_max = 1e16f;

namespace ocarina {
struct alignas(16) Ray {
public:
    float4 org_min{0.f};
    float4 dir_max{0.f};

public:
    explicit Ray(float t_max = ray_t_max)
        : dir_max(make_float4(0, 0, 0, t_max)) {}

    Ray(const float3 origin, const float3 direction,
        float t_max = ray_t_max) noexcept {
        update_origin(origin);
        update_direction(direction);
        dir_max.w = t_max;
    }

    void update_origin(float3 origin) noexcept {
        org_min.x = origin.x;
        org_min.y = origin.y;
        org_min.z = origin.z;
    }

    void update_direction(float3 direction) noexcept {
        dir_max.x = direction.x;
        dir_max.y = direction.y;
        dir_max.z = direction.z;
    }

    [[nodiscard]] auto origin() const noexcept {
        return org_min.xyz();
    }

    [[nodiscard]] auto direction() const noexcept {
        return dir_max.xyz();
    }

    [[nodiscard]] auto at(float t) const noexcept {
        return origin() + direction() * t;
    }

    [[nodiscard]] auto t_max() const noexcept {
        return dir_max.w;
    }

    [[nodiscard]] auto t_min() const noexcept {
        return org_min.w;
    }
};
}// namespace ocarina

OC_STRUCT(ocarina::Ray, org_min, dir_max) {

    [[nodiscard]] auto origin() const noexcept {
        return org_min.xyz();
    }

    [[nodiscard]] auto direction() const noexcept {
        return dir_max.xyz();
    }

    [[nodiscard]] auto at(Float t) const noexcept {
        return origin() + direction();
    }

    [[nodiscard]] auto t_max() const noexcept {
        return dir_max.w;
    }

    [[nodiscard]] auto t_min() const noexcept {
        return org_min.w;
    }
};

namespace ocarina {
struct Triangle {
public:
    uint i, j, k;
    Triangle(uint i, uint j, uint k) : i(i), j(j), k(k) {}
    Triangle() = default;
};
}// namespace ocarina
OC_STRUCT(ocarina::Triangle, i, j, k){};
