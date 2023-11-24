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
[[nodiscard]] Var<bool> over_boundary(Var<int> index, Var<int> size, const string &desc,
                                      const string &tb) noexcept {
    Bool ret = false;
    if_(index > size, [&] {
        $warn(desc + " {}", index);
        ret = true;
    });
    return ret;
}
}// namespace detail

}// namespace ocarina