//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "rhi/descriptor_set.h"
#include <vulkan/vulkan.h>


namespace ocarina {
class VulkanShader;
class VulkanDescriptor;
class VulkanDevice;
class VulkanDescriptorSet;
class VulkanDescriptorSetWriter : public DescriptorSetWriter {
public:
    VulkanDescriptorSetWriter(VulkanDevice* device, VulkanShader **shaders, uint32_t shader_count, VulkanDescriptorSet* descriptor_set);
    ~VulkanDescriptorSetWriter() override = default;
    void bind_buffer(int32_t name_id, handle_ty buffer) override;
    void bind_texture(int32_t name_id, handle_ty texture) override;
    void build();

private:
    std::unordered_map<uint32_t, VulkanDescriptor*> descriptors_;
};

}// namespace ocarina