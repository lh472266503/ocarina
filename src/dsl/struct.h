//
// Created by Zero on 15/05/2022.
//

#pragma once

#include "ast/type.h"



/// make struct ref

/// make struct expr

/// make struct extension

#define KTN_STRUCT(S, ...) \
    KTN_MAKE_STRUCT_REFLECTION(S, ##__VA_ARGS__)