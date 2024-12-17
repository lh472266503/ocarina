//
// Created by Zero on 06/08/2022.
//

#include "vulkan_shader.h"
#include "util.h"
#include "vulkan_device.h"
#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#include "dxcapi.h"

#include <wrl/client.h>
using namespace Microsoft::WRL;
#endif
#include "dxc_compiler.h"

namespace ocarina {

VulkanShader::VulkanShader(VulkanDevice *device, std::span<uint32_t> shaderCode, const std::string_view &entryPoint) : entry_(entryPoint) {
    VkShaderModuleCreateInfo moduleCreateInfo{};
    moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleCreateInfo.codeSize = shaderCode.size() * sizeof(uint32_t);
    moduleCreateInfo.pCode = (uint32_t *)shaderCode.data();
    vkCreateShaderModule(device->logicalDevice(), &moduleCreateInfo, nullptr, &shader_module_);


}

VulkanShader::~VulkanShader() {
    
}

VulkanShader* VulkanShader::create(Device::Impl* device,
                                   ShaderType shader_type, 
    std::span<uint32_t> shaderCode,
    const std::string_view& entryPoint)
{
    return ocarina::new_with_allocator<VulkanShader>(static_cast<VulkanDevice*>(device), shaderCode, entryPoint);
}

VulkanShader *VulkanShader::create_from_HLSL(Device::Impl *device, ShaderType shader_type, const std::string &filename, const std::string &entryPoint) {
    std::ifstream is(filename.data());
    VulkanShader *vulkan_shader = nullptr;
    if (is.is_open()) {
        is.seekg(0, std::ios::end);
        size_t size = is.tellg();
        is.seekg(0, std::ios::beg);
        char *shaderCode = new char[size + 1];
        memset(shaderCode, 0, size + 1);
        //std::vector<char> shader_code(size);
        //is.read(shader_code.data(), size);
        is.read(shaderCode, size);

        is.close();

        assert(size > 0);

        VkShaderStageFlagBits stage = VulkanShader::convert_vulkan_shader_stage(shader_type);
        std::string flattern_hlsl;
        std::string full_directory = get_file_directory(filename);
        std::vector<std::string> include_paths;
        
        
        if (DXCCompiler::preprocess(shaderCode, size, filename, stage, include_paths, flattern_hlsl))
        {
            std::vector<uint32_t> spriv_code;
            std::string errors;
            //bool compiled = HLSLToSPRIV(shader_code, stage, entryPoint, true, spriv_code, errors);

        }

        CompileInput compile_input{
            .hlsl = shaderCode,
            .entry = entryPoint,
            .full_file_path = filename,
            .shader_type = shader_type,
            .output_pdbs = false,
        };
        delete[] shaderCode;
        CompileResult compile_result;
        bool compiled = DXCCompiler::compile_hlsl_spriv(compile_input, compile_result);

        vulkan_shader = create(device, shader_type, compile_result.spriv_codes, entryPoint);

        
    } 

    return vulkan_shader;
}

bool VulkanShader::HLSLToSPRIV(std::span<char> hlsl, VkShaderStageFlagBits stage, const std::string_view &entryPoint, bool outputSymbols, 
    std::vector<uint32_t> &outSpriv, std::string &errorLog) {
    ComPtr<IDxcUtils> dxc_utils = {};
    ComPtr<IDxcCompiler3> dxc_compiler = {};
    
    DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(dxc_utils.ReleaseAndGetAddressOf()));
    DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(dxc_compiler.ReleaseAndGetAddressOf()));

    std::vector<LPCWSTR> args;
    args.push_back(DXC_ARG_PACK_MATRIX_COLUMN_MAJOR);
    args.push_back(L"-HV");
    args.push_back(L"2021");
    args.push_back(L"-T");
    if (stage == VK_SHADER_STAGE_VERTEX_BIT)
    {
        args.push_back(L"vs_6_0");
    }
    else if (stage == VK_SHADER_STAGE_FRAGMENT_BIT)
    {
        args.push_back(L"ps_6_0");
    }
    else if (stage == VK_SHADER_STAGE_COMPUTE_BIT)
    {
        args.push_back(L"cs_6_0");
    }

    std::wstring wEntry(entryPoint.begin(), entryPoint.end());
    args.push_back(L"-E");
    args.push_back(wEntry.c_str());

    if (outputSymbols)
    {
        args.push_back(L"-Zi");
    }

    args.push_back(L"-spirv");
    args.push_back(L"-fspv-target-env=vulkan1.1");

    DxcBuffer src_buffer = {
        hlsl.data(),
        hlsl.size(),
        0};

    IDxcIncludeHandler* dxcIncludeHandler = nullptr;

    ComPtr<IDxcResult> operationResult;
    HRESULT hr = dxc_compiler->Compile(&src_buffer, args.data(), args.size(), dxcIncludeHandler, IID_PPV_ARGS(&operationResult));

    ComPtr<IDxcBlob> shader_obj;
    operationResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shader_obj), nullptr);

    ComPtr<IDxcBlobUtf8> errors = nullptr;
    operationResult->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);

    if (errors != nullptr && errors->GetStringLength() > 0)
    {
        errorLog = errors->GetStringPointer();
        return false;
    }

    outSpriv.resize(shader_obj->GetBufferSize() / sizeof(uint32_t));

    for (size_t i = 0; i < outSpriv.size(); ++i)
    {
        uint32_t spvCode = static_cast<uint32_t *>(shader_obj->GetBufferPointer())[i];
        outSpriv[i] = spvCode;
    }

    return true;
}

}// namespace ocarina