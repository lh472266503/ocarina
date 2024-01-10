//
// Created by Zero on 2023/12/3.
//

#pragma once

#include "function.h"

namespace ocarina {

class FunctionCorrector {
private:
    Function *_function{};

public:
    explicit FunctionCorrector(Function *func) : _function(func) {}

    void traverse(Function &function) noexcept;
};

}// namespace ocarina