//
// Created by Zero on 10/08/2022.
//

#pragma once

#include "util.h"

namespace ocarina {

class CUDADevice;

class Function;

class CUDACompiler {
private:
    CUDADevice *_device;
    const Function &_function;

public:
    CUDACompiler(CUDADevice *device, const Function &f);
    [[nodiscard]] ocarina::string compile(const string &cu, const string &fn) const noexcept;
    [[nodiscard]] ocarina::string obtain_ptx() const noexcept;
};

}// namespace ocarina
