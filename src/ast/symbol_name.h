//
// Created by Zero on 30/07/2022.
//

#pragma once

#include "variable.h"

namespace ocarina {

[[nodiscard]] inline string_view variable_prefix(Variable::Tag tag) {
    using Tag = Variable::Tag;
    switch (tag) {
        case Tag::LOCAL:
            return "v";
        case Tag::DISPATCH_IDX:
            return "d_idx";
        case Tag::DISPATCH_ID:
            return "d_id";
        case Tag::DISPATCH_DIM:
            return "d_dim";
        case Tag::THREAD_IDX:
            return "t_idx";
        case Tag::THREAD_ID:
            return "t_id";
        case Tag::BLOCK_IDX:
            return "b_idx";
        case Tag::BUFFER:
            return "b";
        case Tag::TEXTURE:
            return "t";
        default:
            break;
    }
    OC_ASSERT(0);
    return "";
}

}// namespace ocarina