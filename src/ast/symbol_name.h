//
// Created by Zero on 30/07/2022.
//

#pragma once

#include "variable.h"

namespace ocarina::detail {

template<typename T>
[[nodiscard]] ocarina::string to_string(T &&t) noexcept {
    if constexpr (std::is_same_v<bool, std::remove_cvref_t<T>>) {
        return t ? "true" : "false";
    } else if constexpr (std::is_same_v<float, std::remove_cvref_t<T>>) {
        return ocarina::to_string(std::forward<T>(t)) + "f";
    }
    return ocarina::to_string(std::forward<T>(t));
}

[[nodiscard]] inline string struct_name(uint64_t hash) {
    return "structure_" + to_string(hash);
}

[[nodiscard]] inline string func_name(uint64_t hash) {
    return "function_" + to_string(hash);
}

[[nodiscard]] inline string kernel_name(uint64_t hash) {
    return "kernel_" + to_string(hash);
}

[[nodiscard]] inline string raygen_name(uint64_t hash) {
    return "__raygen__" + to_string(hash);
}

[[nodiscard]] inline string member_name(uint index) {
    return "m" + to_string(index);
}

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
        case Tag::ACCEL:
            return "acc";
        default:
            break;
    }
    OC_ASSERT(0);
    return "";
}

}// namespace ocarina::detail