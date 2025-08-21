//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "vulkan_descriptorset_writer.h"
#include "vulkan_shader.h"
#include "vulkan_buffer.h"
#include "vulkan_descriptorset.h"
#include "vulkan_device.h"
#include "vulkan_driver.h"

namespace ocarina {
VulkanDescriptorSetWriter::VulkanDescriptorSetWriter(VulkanDevice *device, VulkanDescriptorSet *descriptor_set) 
    : descriptor_set_(descriptor_set) {
    VulkanDescriptorSetLayout *layout = descriptor_set->layout();
    size_t bindings_count = layout->get_bindings_count();
    for (size_t i = 0; i < bindings_count; ++i)
    {
        VulkanShaderVariableBinding binding = layout->get_binding(i);
        if (binding.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
            VulkanBuffer *buffer = ocarina::new_with_allocator<VulkanBuffer>(device, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, binding.size);
            // Create a descriptor for uniform buffer
            VulkanDescriptorBuffer *descriptor_buffer = ocarina::new_with_allocator<VulkanDescriptorBuffer>();
            descriptor_buffer->binding = binding.binding;
            descriptor_buffer->name_ = binding.name;
            descriptor_buffer->buffer_ = buffer;
            bind_buffer(binding.binding, buffer->get_descriptor_info());
            buffers_.insert(std::make_pair(binding.binding, buffer));
            descriptors_.insert(std::make_pair(hash64(descriptor_buffer->name_), descriptor_buffer));
        }
        // Add other types of descriptors as needed
    }

    build(device);
}

VulkanDescriptorSetWriter::~VulkanDescriptorSetWriter()
{
    for (auto& buffer : buffers_)
    {
        if (buffer.second) {
            ocarina::delete_with_allocator(buffer.second);
        }
    }
    buffers_.clear();
    for (auto &descriptor : descriptors_) {
        if (descriptor.second) {
            if (descriptor.second->is_buffer_) {
                VulkanDescriptorBuffer *buffer_descriptor = static_cast<VulkanDescriptorBuffer *>(descriptor.second);
                ocarina::delete_with_allocator(buffer_descriptor);
            }
           
        }
    }
    descriptors_.clear();
}

void VulkanDescriptorSetWriter::bind_buffer(uint32_t binding, VkDescriptorBufferInfo* buffer)
{
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set_->descriptor_set();
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = buffer;
    writes_.push_back(write);
}

//void VulkanDescriptorSetWriter::bind_buffer(uint64_t name_id, handle_ty buffer) {
//
//}
//
//void VulkanDescriptorSetWriter::bind_texture(uint64_t name_id, handle_ty texture) {
//
//}

void VulkanDescriptorSetWriter::build(VulkanDevice *device) {
    
    std::vector<VkWriteDescriptorSet> writes;

    /*
    for (auto &descriptor : pending_writes_) {
        if (descriptor->is_buffer_) {
            VulkanDescriptorBuffer *buffer_descriptor = static_cast<VulkanDescriptorBuffer *>(descriptor);
            VkDescriptorBufferInfo* buffer_info = buffer_descriptor->buffer_->get_descriptor_info();

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptor_set_->descriptor_set();
            write.dstBinding = buffer_descriptor->binding;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.pBufferInfo = buffer_info;
            writes.push_back(write);
        }
    }
    */
    for (auto &write : writes_) {
        writes.push_back(write);
    }

    vkUpdateDescriptorSets(device->logicalDevice(), writes.size(), writes.data(), 0, nullptr);
    pending_writes_.clear();
}

void VulkanDescriptorSetWriter::update_buffer(uint64_t name_id, void *data, uint32_t size) {
    auto it = descriptors_.find(name_id);
    if (it != descriptors_.end()) {
        VulkanDescriptorBuffer *descriptor_buffer = static_cast<VulkanDescriptorBuffer *>(it->second);
        descriptor_buffer->buffer_->load_from_cpu(data, 0, size);
    }
}

void VulkanDescriptorSetWriter::update_push_constants(uint64_t name_id, void *data, uint32_t size, Pipeline *pipeline) {
    auto it = descriptors_.find(name_id);
    if (it != descriptors_.end()) {
        VulkanDescriptorPushConstants *push_constants_descriptor = static_cast<VulkanDescriptorPushConstants *>(it->second);
        // Assuming push constants are handled in a specific way
        // This is a placeholder for actual push constant update logic
        VulkanPipeline *vulkan_pipeline = static_cast<VulkanPipeline *>(pipeline);
        VkPipelineLayout layout = vulkan_pipeline->pipeline_layout_;
        VulkanDriver::instance().push_constants(layout, data, size, 0);
    }
}

}// namespace ocarina