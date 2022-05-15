//
// Created by Zero on 15/05/2022.
//

#pragma once
#include "core/basic_traits.h"
#include "ast/type_registry.h"
#include "core/macro_map.h"

namespace katana {

struct alignas(16) Hit {
    uint inst_id{};
    uint prim_id{};
    float2 bary;
};

template<>
struct is_struct<Hit> : std::true_type {};

template<>
struct struct_member_tuple<Hit> {
    using this_type = Hit;
    using type = std::tuple<std::remove_cvref_t<decltype(this_type::inst_id)>, std::remove_cvref_t<decltype(this_type::prim_id)>, std::remove_cvref_t<decltype(this_type::bary)>>;
    using offset = std::index_sequence<KTN_OFFSET_OF(this_type, inst_id), KTN_OFFSET_OF(this_type, prim_id), KTN_OFFSET_OF(this_type, bary)>;
    static_assert(is_valid_reflection_v<this_type, type, offset>, "may be order of members is wrong!");
};

}