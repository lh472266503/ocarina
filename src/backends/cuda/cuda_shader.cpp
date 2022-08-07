//
// Created by zero on 2022/7/18.
//

#include "cuda_shader.h"
#include "util.h"
#include "cuda_device.h"

namespace ocarina {

CUDAShader::CUDAShader(Device::Impl *device,
                       const ocarina::string &ptx,
                       const Function &func)
    : _device(dynamic_cast<CUDADevice *>(device)),
      _function(func) {
    OC_CU_CHECK(cuModuleLoadData(&_module, ptx.c_str()));
    OC_CU_CHECK(cuModuleGetFunction(&_func_handle, _module, _function.func_name().c_str()));
#if CUDA_ARGUMENT_PUSH == 0
    _function.for_each_uniform_var([&](const UniformBinding &uniform) {
        const string &var_name = uniform.expression()->variable().name();
        CUdeviceptr ptr{};
        size_t size{};
        OC_CU_CHECK(cuModuleGetGlobal(&ptr, &size, _module, var_name.c_str()));
        OC_CU_CHECK(cuMemcpyHtoD(ptr, uniform.handle_address(), size));
    });
#endif
}

CUDAShader::~CUDAShader() {
    OC_CU_CHECK(cuModuleUnload(_module));
}

void CUDAShader::launch(handle_ty stream, ShaderDispatchCommand *cmd) noexcept {

    uint3 grid_dim = make_uint3(1);
    uint3 block_dim = make_uint3(1);
    if (_function.has_configure()) {
        grid_dim = _function.grid_dim();
        block_dim = _function.block_dim();
    } else {
        grid_dim = (cmd->dispatch_dim() + block_dim - 1u) / block_dim;
    }
    OC_CU_CHECK(cuLaunchKernel(_func_handle, grid_dim.x, grid_dim.y, grid_dim.z,
                               block_dim.x, block_dim.y, block_dim.z,
                               0, reinterpret_cast<CUstream>(stream), cmd->args().data(), nullptr));
}

void CUDAShader::compute_fit_size() noexcept {
    _device->use_context([&] {
        int min_grid_size;
        int auto_block_size;
        OC_CU_CHECK(cuOccupancyMaxPotentialBlockSize(&min_grid_size, &auto_block_size,
                                                     _func_handle, 0, 0, 0));

        _function.set_grid_dim(min_grid_size);
        _function.set_block_dim(auto_block_size);
    });
}

}// namespace ocarina