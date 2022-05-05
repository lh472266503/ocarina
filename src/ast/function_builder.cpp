//
// Created by Zero on 03/05/2022.
//

#include "function_builder.h"

namespace sycamore::ast {
sycamore::vector<FunctionBuilder *> &FunctionBuilder::_function_stack() noexcept {
    static thread_local sycamore::vector<FunctionBuilder *> stack;
    return stack;
}


}// namespace sycamore::ast