//
// Created by Zero on 15/05/2022.
//

#pragma once

#include "core/basic_traits.h"
#include "core/macro_map.h"

/// make struct reflection
#define KTN_MEMBER_TYPE_MAP(member) std::remove_cvref_t<decltype(this_type::member)>
#define KTN_TYPE_OFFSET_OF(member) KTN_OFFSET_OF(this_type, member)
#define KTN_MAKE_STRUCT_REFLECTION(S, ...)                                               \
    template<>                                                                           \
    struct is_struct<S> : std::true_type {};                                             \
    template<>                                                                           \
    struct struct_member_tuple<S> {                                                      \
        using this_type = Hit;                                                           \
        using type = std::tuple<MAP_LIST(KTN_MEMBER_TYPE_MAP, ##__VA_ARGS__)>;           \
        using offset = std::index_sequence<MAP_LIST(KTN_TYPE_OFFSET_OF, ##__VA_ARGS__)>; \
        static_assert(is_valid_reflection_v<this_type, type, offset>,                    \
                      "may be order of members is wrong!");                              \
    };

/// make struct type description

/// make struct ref

/// make struct expr

/// make struct extension

#define KTN_STRUCT(S, ...) \
    KTN_MAKE_STRUCT_REFLECTION(S, ##__VA_ARGS__)