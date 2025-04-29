#pragma once
#include "core/header.h"
#include "core/concepts.h"
#include "core/stl.h"
#include "core/singleton.h"
#include "rhi/index_buffer.h"
#include <vulkan/vulkan.h>
#include "vulkan_buffer.h"
namespace ocarina {
class VulkanDevice;

class VulkanIndexBuffer : public IndexBuffer {
public:
    VulkanIndexBuffer(VulkanDevice *device, uint32_t size);
    ~VulkanIndexBuffer();
    void load_from_cpu(const void *cpuData, uint32_t byteOffset,
                     uint32_t numBytes);

    VkBuffer buffer_handle() const
    {
        return vulkan_buffer_ != nullptr ? vulkan_buffer_->buffer_handle() : VK_NULL_HANDLE;
    }
private:
    VulkanDevice *device_ = nullptr;
    VulkanBuffer* vulkan_buffer_ = nullptr;

};

}// namespace ocarina