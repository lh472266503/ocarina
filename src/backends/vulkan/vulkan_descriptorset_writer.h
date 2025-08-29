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
class VulkanBuffer;
class VulkanDescriptorSetWriter : public DescriptorSetWriter {
public:
    VulkanDescriptorSetWriter(VulkanDevice* device, VulkanDescriptorSet* descriptor_set);
    ~VulkanDescriptorSetWriter();
    void bind_buffer(uint32_t binding, VkDescriptorBufferInfo* buffer);
    void bind_texture(uint32_t binding, VkDescriptorImageInfo* texture);
    void build(VulkanDevice* device);

    void update_buffer(uint64_t name_id, void *data, uint32_t size) override;
    void update_push_constants(uint64_t name_id, void *data, uint32_t size, Pipeline* pipeline) override;
    void update_texture(uint64_t name_id, Texture *texture) override;

private:
    std::unordered_map<uint64_t, VulkanDescriptor*> descriptors_;
    std::unordered_map<uint32_t, VulkanBuffer*> buffers_;
    std::vector<VulkanDescriptor *> pending_writes_;
    std::vector<VkWriteDescriptorSet> writes_;
    VulkanDescriptorSet *descriptor_set_ = nullptr;
};

}// namespace ocarina