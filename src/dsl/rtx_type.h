//
// Created by Zero on 27/07/2022.
//

#pragma once

#include "core/basic_traits.h"
#include "dsl/operators.h"
#include "dsl/struct.h"
#include "computable.h"

namespace ocarina {

struct alignas(16) Hit {
    uint inst_id{uint(-1)};
    uint prim_id{uint(-1)};
    float2 bary;
};

}// namespace ocarina

// clang-format off
OC_STRUCT(ocarina::Hit, inst_id, prim_id, bary){
    void init(){
        inst_id = uint(-1);
    }
    [[nodiscard]] Bool is_miss() const noexcept {
        return eval(inst_id == uint(-1));
    }
    [[nodiscard]] Bool is_hit() const noexcept {
        return !is_miss();
    }
    template<typename... Args>
    [[nodiscard]] auto lerp(Args &&...args) const noexcept {
        return ocarina::triangle_lerp(bary, OC_FORWARD(args)...);
    }
};
// clang-format on

namespace ocarina {
using OCHit = Var<Hit>;
}

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

    [[nodiscard]] auto origin() const noexcept { return org_min.xyz(); }
    [[nodiscard]] auto direction() const noexcept { return dir_max.xyz(); }
    [[nodiscard]] auto at(float t) const noexcept { return origin() + direction() * t; }
    [[nodiscard]] auto t_max() const noexcept { return dir_max.w; }
    [[nodiscard]] auto t_min() const noexcept { return org_min.w; }
};

template<typename... Args>
requires none_dsl_v<Args...>
[[nodiscard]] Ray make_ray(Args &&...args) noexcept {
    return Ray{OC_FORWARD(args)...};
}

}// namespace ocarina

// clang-format off
OC_STRUCT(ocarina::Ray, org_min, dir_max) {

    void update_origin(Float3 origin) noexcept {
        org_min.x = origin.x;
        org_min.y = origin.y;
        org_min.z = origin.z;
    }

    void update_direction(Float3 direction) noexcept {
        dir_max.x = direction.x;
        dir_max.y = direction.y;
        dir_max.z = direction.z;
    }

    [[nodiscard]] auto origin() const noexcept { return org_min.xyz(); }
    [[nodiscard]] auto direction() const noexcept { return dir_max.xyz(); }
    [[nodiscard]] auto at(Float t) const noexcept { return origin() + direction() * t; }
    [[nodiscard]] auto t_max() const noexcept { return dir_max.w; }
    [[nodiscard]] auto t_min() const noexcept { return org_min.w; }
};
// clang-format on

namespace ocarina {
using OCRay = Var<Ray>;
}

namespace ocarina {

inline float3 offset_ray_origin(const float3 &p_in, const float3 &n_in) noexcept {
    constexpr auto origin = 1.0f / 32.0f;
    constexpr auto float_scale = 1.0f / 65536.0f;
    constexpr auto int_scale = 256.0f;
    float3 n = n_in;
    auto of_i = make_int3(static_cast<int>(int_scale * n.x),
                          static_cast<int>(int_scale * n.y),
                          static_cast<int>(int_scale * n.z));
    float3 p = p_in;
    float3 p_i = make_float3(
        bit_cast<float>(bit_cast<int>(p.x) + select(p.x < 0, -of_i.x, of_i.x)),
        bit_cast<float>(bit_cast<int>(p.y) + select(p.y < 0, -of_i.y, of_i.y)),
        bit_cast<float>(bit_cast<int>(p.z) + select(p.z < 0, -of_i.z, of_i.z)));
    return select(abs(p) < origin, p + float_scale * n, p_i);
}

}// namespace ocarina
