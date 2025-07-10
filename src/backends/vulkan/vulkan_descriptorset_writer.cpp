//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "vulkan_descriptorset_writer.h"
#include "vulkan_shader.h"
#include "vulkan_buffer.h"
#include "vulkan_descriptorset.h"

namespace ocarina {


VulkanDescriptorSetWriter::VulkanDescriptorSetWriter(VulkanDevice *device, VulkanShader **shaders, uint32_t shader_count, VulkanDescriptorSet *descriptor_set) {
    for (size_t i = 0; i < shader_count; ++i) {
        VulkanShader* shader = shaders[i];
        uint32_t variable_count = shader->get_shader_variables_count();
        for (uint32_t j = 0; j < variable_count; ++j) {
            const VulkanShaderVariableBinding &variable = shader->get_shader_variable(j);
            if (variable.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
                //// Create a descriptor for uniform buffer
                //VulkanDescriptorBuffer *descriptor = ocarina::new_with_allocator<VulkanDescriptorBuffer>();
                //descriptor->binding = variable.binding;
                //descriptor->name_ = variable.name;
                //descriptor->buffer_ = nullptr;// Will be set later
                //descriptors_[variable.binding] = descriptor;
                VulkanBuffer *buffer = ocarina::new_with_allocator<VulkanBuffer>(device, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, variable.size);

                VulkanDescriptor *descriptor = ocarina::new_with_allocator<VulkanDescriptorBuffer>();
            } else if (variable.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {

            }
            // Add other types of descriptors as needed
        }
        
    }
}

void VulkanDescriptorSetWriter::bind_buffer(int32_t name_id, handle_ty buffer){

}

void VulkanDescriptorSetWriter::bind_texture(int32_t name_id, handle_ty texture) {

}

}// namespace ocarina