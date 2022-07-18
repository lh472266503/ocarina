//
// Created by Zero on 2022/7/15.
//

#include "cuda_codegen.h"

namespace ocarina {

void CUDACodegen::_emit_function(const Function &f) noexcept {
    if (f.has_defined()) {
        return;
    }
    switch (f.tag()) {
        case Function::Tag::KERNEL: _scratch << "extern \"C\" __global__ "; break;
        case Function::Tag::CALLABLE: _scratch << "__device__ "; break;
    }
    CppCodegen::_emit_function(f);
}
}// namespace ocarina