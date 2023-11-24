//
// Created by Zero on 2023/11/23.
//

#include "dsl.h"

namespace ocarina {

Env *Env::s_env = nullptr;

Env &Env::instance() noexcept {
    if (s_env == nullptr) {
        s_env = new Env();
    }
    return *s_env;
}

void Env::destroy_instance() noexcept {
    if (s_env) {
        delete s_env;
        s_env = nullptr;
    }
}

namespace detail {
[[nodiscard]] Var<uint> correct_index(Var<uint> index, Var<uint> size, const string &desc,
                                      const string &tb) noexcept {
    if (Env::valid_check()) {
        if_(index >= size, [&] {
            string tips = ocarina::format("buffer access over boundary : {}, ", desc.c_str());
            $warn_with_location(tips + "index = {}, size = {}, current thread will be terminated \n" + tb, index, size);
            index = 0;
            $return();
        });
    }
    return index;
}

[[nodiscard]] Var<uint> divide(Var<uint> lhs, Var<uint> rhs) noexcept {
    return lhs / rhs;
}

}// namespace detail

}// namespace ocarina