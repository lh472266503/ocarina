//
// Created by Zero on 2023/12/3.
//

#include "function_corrector.h"

namespace ocarina {

void FunctionCorrector::traverse(Function &function) noexcept {
}

void FunctionCorrector::apply() noexcept {
    traverse(*_function);
}

}// namespace ocarina