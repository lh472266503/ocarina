//
// Created by Zero on 03/05/2022.
//

#include "function_builder.h"

namespace katana::ast {
katana::vector<FunctionBuilder *> &FunctionBuilder::_function_stack() noexcept {
    static thread_local katana::vector<FunctionBuilder *> stack;
    return stack;
}
FunctionBuilder *FunctionBuilder::current() noexcept {
    return _function_stack().back();
}

}// namespace katana::ast