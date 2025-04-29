#include "util.h"
#include "vulkan_device.h"
#include "vulkan_buffer.h"

namespace ocarina {

VulkanBuffer::VulkanBuffer(VulkanDevice *device, VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags memory_property_flags, VkDeviceSize size, const void *data ) : device_(device) {
    memory_property_flags_ = memory_property_flags;
    VkBufferCreateInfo buffer_create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_create.usage = usage_flags;
    buffer_create.size = size;
    vkCreateBuffer(device->logicalDevice(), &buffer_create, nullptr, &vulkan_buffer_);

    if (data)
    {
        load_from_cpu(data, 0, size);
    }
}

VulkanBuffer::~VulkanBuffer() {
    if (vulkan_buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_->logicalDevice(), vulkan_buffer_, nullptr);
    }

    if (memory_ != VK_NULL_HANDLE)
    {
        vkFreeMemory(device_->logicalDevice(), memory_, nullptr);
    }
}

void VulkanBuffer::load_from_cpu(const void* cpuData, uint32_t byteOffset,
    uint32_t numBytes)
{
    // Create the memory backing up the buffer handle
    VkMemoryRequirements memReqs;
    VkMemoryAllocateInfo memAlloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};

    vkGetBufferMemoryRequirements(device_->logicalDevice(), vulkan_buffer_, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    // Find a memory type index that fits the properties of the buffer
    memAlloc.memoryTypeIndex = device_->get_memory_type(memReqs.memoryTypeBits, memory_property_flags_);
    // If the buffer has VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT set we also need to enable the appropriate flag during allocation
    VkMemoryAllocateFlagsInfoKHR allocFlagsInfo{};
    if (usage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR;
        allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
        memAlloc.pNext = &allocFlagsInfo;
    }
    VK_CHECK_RESULT(vkAllocateMemory(device_->logicalDevice(), &memAlloc, nullptr, &memory_));

    // If a pointer to the buffer data has been passed, map the buffer and copy over the data
    if (cpuData != nullptr) {
        VK_CHECK_RESULT(vkMapMemory(device_->logicalDevice(), memory_, byteOffset, numBytes, 0, &mapped_));
        memcpy(mapped_, cpuData, numBytes);
        if ((memory_property_flags_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
            flush();

        if (mapped_) {
            vkUnmapMemory(device_->logicalDevice(), memory_);
            mapped_ = nullptr;
        }
    }
}

VkResult VulkanBuffer::flush(VkDeviceSize size, VkDeviceSize offset) {
    if (mapped_ == nullptr) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory_;
    range.offset = offset;
    range.size = size;

    return vkFlushMappedMemoryRanges(device_->logicalDevice(), 1, &range);
}

VulkanBuffer *VulkanBufferManager::create_vulkan_buffer(VulkanDevice *device, VkBufferUsageFlags usage_flags, 
    VkMemoryPropertyFlags memory_property_flags, VkDeviceSize size, const void *initial_data) {
    VulkanBuffer *vulkan_buffer = new VulkanBuffer(device, usage_flags, memory_property_flags, size, initial_data);

    if (vulkan_buffer->buffer_handle() != VK_NULL_HANDLE) {
        vulkan_buffers.insert({vulkan_buffer->buffer_handle(), vulkan_buffer});
    } else {
        delete vulkan_buffer;
        return nullptr;
    }
    

    return vulkan_buffer;
}

VulkanBuffer *VulkanBufferManager::get_vulkan_buffer(VkBuffer handle) {
    auto it = vulkan_buffers.find(handle);

    if (it != vulkan_buffers.end()) {
        return it->second;
    }

    return nullptr;
}

void VulkanBufferManager::clear() {
    for (auto& buffer : vulkan_buffers) {
        delete buffer.second;
    }
    vulkan_buffers.clear();
}

}// namespace ocarina


