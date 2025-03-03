//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include "rhi/resources/shader.h"
#include <vulkan/vulkan.h>

namespace ocarina {

    class VulkanDevice;

class VulkanShader : public Shader<>::Impl {
private:
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    std::string entry_;
    VulkanDevice *device_ = nullptr;
    VkShaderStageFlagBits stage_;

    static bool HLSLToSPRIV(std::span<char> hlsl, VkShaderStageFlagBits stage, const std::string_view &entryPoint, bool outputSymbols, std::vector<uint32_t> &outSpriv, std::string &errorLog);

public:
    VulkanShader(VulkanDevice *device, std::span<uint32_t> shaderCode, const std::string_view &entryPoint, VkShaderStageFlagBits stage);
    ~VulkanShader() override;

    OC_MAKE_MEMBER_GETTER(shader_module, );
    OC_MAKE_MEMBER_GETTER(stage, );

    const char* get_entry_point() const
    {
        return entry_.c_str();
    }
    void launch(handle_ty stream, ShaderDispatchCommand *cmd) noexcept override {}
    static VulkanShader *create(Device::Impl *device,
                                ShaderType shader_type, 
                                std::span<uint32_t> shader_code, 
                                const std::string_view &entry_point);

    static VulkanShader *create_from_HLSL(Device::Impl *device, 
        ShaderType shader_type, 
        const std::string& filename, 
        const std::string& entry_point);

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

struct ShaderKey {
    ShaderType shader_type = ShaderType::VertexShader;
    std::string shader_code;
    std::string entry_point;
    std::set<std::string> options;

    bool operator==(const ShaderKey &other) const {
        return shader_type == other.shader_type && entry_point == other.entry_point && shader_code == other.shader_code && options == other.options;
    }
};

struct VulkanShaderEntry
{
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkShaderStageFlagBits stage;
    const char *entry = nullptr;
    bool is_valid() const
    {
        return shader_module != VK_NULL_HANDLE;
    }
};

struct HashShaderKeyFunction {
    uint64_t operator()(const ShaderKey &shader_key) const {
        std::size_t res = 0;
        hash_combine(res, *((uint64_t *)&shader_key.shader_type));
        hash_combine(res, std::hash<std::string>()(shader_key.shader_code));
        hash_combine(res, std::hash<std::string>()(shader_key.entry_point));
        for (auto iter : shader_key.options) {
            hash_combine(res, std::hash<std::string>()(iter));
        }
        return res;
    }
};

class VulkanShaderManager : concepts::Noncopyable {
public:
    //VulkanShader *get_or_create_from_HLSL(Device::Impl *device,
    //                               ShaderType shader_type,
    //                               const std::string &filename,
    //                               const std::string &entry_point);
    handle_ty get_or_create_from_HLSL(Device::Impl *device,
                                      ShaderType shader_type,
                                      const std::string &filename,
                                      const std::string &entry_point);

    VulkanShaderEntry get_shader_entry(handle_ty shader_handle) const;
    void clear(VulkanDevice *device);
private:
    std::unordered_map<ShaderKey, VulkanShader *, HashShaderKeyFunction> vulkan_shaders_;
    std::map<handle_ty, VulkanShaderEntry> vulkan_shader_entries_;
};
}// namespace ocarina