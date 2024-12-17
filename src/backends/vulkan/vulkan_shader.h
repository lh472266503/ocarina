//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/resources/shader.h"
#include <vulkan/vulkan.h>

namespace ocarina {

    class VulkanDevice;

class VulkanShader : public Shader<>::Impl {
private:
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    std::string entry_;

    static bool HLSLToSPRIV(std::span<char> hlsl, VkShaderStageFlagBits stage, const std::string_view &entryPoint, bool outputSymbols, std::vector<uint32_t> &outSpriv, std::string &errorLog);

public:
    VulkanShader(VulkanDevice *device, std::span<uint32_t> shaderCode, const std::string_view &entryPoint);
    ~VulkanShader() override;

    OC_MAKE_MEMBER_GETTER(shader_module, );

    const char* GetEntryPoint() const
    {
        return entry_.c_str();
    }
    void launch(handle_ty stream, ShaderDispatchCommand *cmd) noexcept override {}
    static VulkanShader *create(Device::Impl *device,
                                ShaderType shader_type, 
                                std::span<uint32_t> shaderCode, 
                                const std::string_view &entryPoint);

    static VulkanShader *create_from_HLSL(Device::Impl *device, 
        ShaderType shader_type, 
        const std::string& filename, 
        const std::string& entryPoint);

    static VkShaderStageFlagBits convert_vulkan_shader_stage(ShaderType shader_type)
    {
        switch (shader_type) {
            case ocarina::ShaderType::VertexShader:
                return VK_SHADER_STAGE_VERTEX_BIT;
                break;
            case ocarina::ShaderType::PixelShader:
                return VK_SHADER_STAGE_FRAGMENT_BIT;
                break;
            case ocarina::ShaderType::GeometryShader:
                return VK_SHADER_STAGE_GEOMETRY_BIT;
                break;
            case ocarina::ShaderType::ComputeShader:
                return VK_SHADER_STAGE_COMPUTE_BIT;
                break;
            case ocarina::ShaderType::MeshShader:
                return VK_SHADER_STAGE_MESH_BIT_EXT;
                break;
            default:
                return VK_SHADER_STAGE_VERTEX_BIT;
                break;
        }
    }
};
}// namespace ocarina