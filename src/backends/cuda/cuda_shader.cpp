//
// Created by zero on 2022/7/18.
//

#include "cuda_shader.h"
#include "util.h"

namespace ocarina {

CUDAShader::CUDAShader(Device::Impl *device,
                       const ocarina::string &ptx,
                       ocarina::string_view entry)
    : _entry(entry) {
    OC_CU_CHECK(cuModuleLoadData(&_module, ptx.c_str()));
    OC_CU_CHECK(cuModuleGetFunction(&_function, _module, _entry.c_str()));
}

void CUDAShader::launch(handle_ty stream, ShaderDispatchCommand *cmd) noexcept {
    uint3 grid_size = make_uint3(1);
    uint3 block_size = cmd->dim();
    OC_CU_CHECK(cuLaunchKernel(_function, grid_size.x, grid_size.y, grid_size.z,
                               block_size.x, block_size.y, block_size.z,
                               0, reinterpret_cast<CUstream>(stream), cmd->args().data(), nullptr));
}

}// namespace ocarina