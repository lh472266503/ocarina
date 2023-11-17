//
// Created by Zero on 2023/11/17.
//

#pragma once

#include "core/basic_types.h"
#include "dsl/type_trait.h"
#include "dsl/computable.h"

namespace ocarina {

class Env {
private:
    Env() = default;
    Env(const Env &) = delete;
    Env(Env &&) = delete;
    Env operator=(const Env &) = delete;
    Env operator=(Env &&) = delete;
    static Env *s_env;

public:

};

}// namespace ocarina