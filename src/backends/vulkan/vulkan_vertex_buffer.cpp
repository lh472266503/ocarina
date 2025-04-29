#pragma once
#include "vulkan_vertex_buffer.h"
#include "vulkan_shader.h"
#include "vulkan_device.h"
#include "vulkan_buffer.h"

namespace ocarina {

void VulkanVertexStreamBinding::create_from_vertex_shader(VulkanShader *vertex_shader, VulkanVertexBuffer *vertex_buffer, VulkanVertexStreamBinding &binding) {
    size_t attr_count = vertex_shader->get_vertex_attribute_count();

    binding.attribute_descriptions_.resize(attr_count);
    binding.binding_descriptions_.resize(attr_count);
    binding.buffers_.resize(attr_count);
    binding.offsets_.resize(attr_count);

    for (size_t i = 0; i < attr_count; ++i)
    {
        auto attr = vertex_shader->get_vertex_attribute(i);
        binding.attribute_descriptions_[i].binding = i;
        binding.attribute_descriptions_[i].location = attr.location;
        binding.attribute_descriptions_[i].format = static_cast<VkFormat>(attr.format);
        binding.attribute_descriptions_[i].offset = attr.offset;

        auto vertex_stream = vertex_buffer->get_vertex_stream((VertexAttributeType::Enum)attr.type);

        assert(vertex_stream != nullptr);

        if (vertex_stream->buffer_handle != InvalidUI64)
        {
            binding.buffers_[i] = (VkBuffer)vertex_stream->buffer_handle;
        }
        else
        {
            VulkanDevice *device = static_cast<VulkanDevice*>(vertex_buffer->device());
            VulkanBuffer* buffer = device->create_vulkan_buffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                vertex_stream->get_size(),
                vertex_stream->data);
            binding.buffers_[i] = buffer->buffer_handle();
            vertex_stream->buffer_handle = (handle_ty)binding.buffers_[i];
        }

        binding.binding_descriptions_[i].binding = i;
        binding.binding_descriptions_[i].stride = vertex_stream->stride;
        binding.binding_descriptions_[i].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    }
}

VulkanVertexBuffer::VulkanVertexBuffer(VulkanDevice *device) {
    device_ = device;
}

VulkanVertexBuffer::~VulkanVertexBuffer()
{
    for (auto it : vertex_bindings_)
    {
        delete it.second;
    }
}

VulkanVertexStreamBinding *VulkanVertexBuffer::get_or_create_vertex_binding(VulkanShader *vertex_shader) {
    auto it = vertex_bindings_.find((handle_ty)vertex_shader->shader_module());

    if (it != vertex_bindings_.end())
    {
        return it->second;
    }

    auto binding = new VulkanVertexStreamBinding();
    VulkanVertexStreamBinding::create_from_vertex_shader(vertex_shader, this, *binding);
    vertex_bindings_.insert(std::make_pair((handle_ty)vertex_shader->shader_module(), binding));

    return binding;
}

void VulkanVertexBuffer::upload_data(VertexAttributeType::Enum type) {

}

}// namespace ocarina