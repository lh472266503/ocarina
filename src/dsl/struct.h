//
// Created by Zero on 15/05/2022.
//

#pragma once

#include "ast/type_registry.h"
#include "var.h"
#include "soa.h"

#define OC_STRUCT(S, ...)                       \
    OC_MAKE_STRUCT_REFLECTION(S, ##__VA_ARGS__) \
    OC_MAKE_STRUCT_DESC(S, ##__VA_ARGS__)       \
    OC_MAKE_COMPUTABLE_BODY(S, ##__VA_ARGS__)   \
    OC_MAKE_PROXY(S)
