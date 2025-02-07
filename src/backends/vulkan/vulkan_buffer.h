#pragma once
#include "core/header.h"
#include "core/concepts.h"
#include "core/stl.h"
#include <vulkan/vulkan.h>
namespace ocarina {
class VulkanDevice;
struct VulkanVertexInfo {
    std::vector<VkVertexInputBindingDescription> binding_descriptions;
    std::vector<VkVertexInputAttributeDescription> attribute_descriptions;
};

class VulkanBuffer {
public:
    VulkanBuffer(VulkanDevice *device);
    ~VulkanBuffer();
    void load_from_cpu(VkCommandBuffer cmdbuf, const void *cpuData, uint32_t byteOffset,
                     uint32_t numBytes);

private:
    VulkanDevice *device_ = nullptr;
    VkBuffer vulkan_buffer_ = VK_NULL_HANDLE;
    VkBufferUsageFlags usage_ = {};
};
}// namespace ocarina