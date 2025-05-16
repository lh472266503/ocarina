
#include "vulkan_device.h"
#include "vulkan_pipeline.h"
#include "util.h"
#include "vulkan_shader.h"
#include "vulkan_vertex_buffer.h"
#include "vulkan_driver.h"

namespace ocarina {

void VulkanPipelineManager::bind_shader(handle_ty shader, int stage) {
    pipeline_key_cache_.shaders[stage] = (VkShaderModule)shader;
}

void VulkanPipelineManager::bind_raster_state(const RasterState& raster_state)
{
    pipeline_key_cache_.raster_state = raster_state;
}

void VulkanPipelineManager::bind_blend_state(const BlendState &blend_state) {
    pipeline_key_cache_.blend_state = blend_state;
}

void VulkanPipelineManager::bind_depth_stencil_state(const DepthStencilState& depth_stencil_state)
{
    pipeline_key_cache_.depth_stencil_state = depth_stencil_state;
}

void VulkanPipelineManager::bind_pipeline_layout(VkPipelineLayout pipeline_layout) {
    pipeline_key_cache_.pipeline_layout = pipeline_layout;
}

void VulkanPipelineManager::bind_vertex_attributes(VkVertexInputAttributeDescription const* attributes,
                                                   VkVertexInputBindingDescription const *binds, uint8_t attr_count, uint8_t bind_desc_count) {
    for (size_t i = 0; i < PipelineKey::MAX_VERTEX_ATTRIBUTES; i++) {
        if (i < attr_count) {
            pipeline_key_cache_.vertex_input_attributes[i] = attributes[i];
        } 
        else {
            pipeline_key_cache_.vertex_input_attributes[i] = {};
        }
    }

    for (size_t i = 0; i < PipelineKey::MAX_VERTEX_ATTRIBUTES; i++) {
        if (i < bind_desc_count) {
            pipeline_key_cache_.vertex_input_binding[i] = binds[i];
        } else {
            pipeline_key_cache_.vertex_input_binding[i] = {};
        }
    }
}

void VulkanPipelineManager::bind_topology(PrimitiveType primitive_type) {
    pipeline_key_cache_.topology = get_vulkan_topology(primitive_type);
}

std::tuple<VkPipelineLayout, VulkanPipeline> VulkanPipelineManager::get_or_create_pipeline(const PipelineState &pipeline_state, VulkanDevice *device) {
    

    for (int i = 0; i < PipelineKey::MAX_SHADER_STAGE; ++i) {
        bind_shader(pipeline_state.shaders[i], i);
    }
    bind_blend_state(pipeline_state.blend_state);
    bind_depth_stencil_state(pipeline_state.depth_stencil_state);
    bind_raster_state(pipeline_state.raster_state);

    VulkanShader *vertex_shader = VulkanDriver::instance().get_shader(pipeline_state.shaders[0]);
    VulkanShader *pixel_shader = VulkanDriver::instance().get_shader(pipeline_state.shaders[1]);
    VulkanShader* compute_shader = VulkanDriver::instance().get_shader(pipeline_state.shaders[2]);

    VulkanVertexBuffer *vertex_buffer = static_cast<VulkanVertexBuffer *>(pipeline_state.vertex_buffer);
    VulkanVertexStreamBinding *vertex_binding = nullptr;
    if (vertex_buffer) {
        vertex_binding = vertex_buffer->get_or_create_vertex_binding(vertex_shader);

        bind_vertex_attributes(vertex_binding->attribute_descriptions_.data(), vertex_binding->binding_descriptions_.data(),
                               vertex_binding->attribute_descriptions_.size(), vertex_binding->binding_descriptions_.size());
    }
    bind_topology(pipeline_state.primitive_type);

    VkDescriptorSetLayout descriptor_set_layouts[4];
    VulkanShader *shaders[2] = { vertex_shader, pixel_shader };
    descriptor_set_layouts[0] = VulkanDriver::instance().create_descriptor_set_layout(shaders, 2);

    pipeline_key_cache_.pipeline_layout = VulkanDriver::instance().get_pipeline_layout(descriptor_set_layouts, 1);

    auto it = vulkan_pipelines_.find(pipeline_key_cache_);
    if (it != vulkan_pipelines_.end()) {
        return std::make_tuple(pipeline_key_cache_.pipeline_layout, it->second);
    }

    VkPipelineVertexInputStateCreateInfo vertex_input_state = {};
    vertex_input_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_state.vertexBindingDescriptionCount = vertex_binding->binding_descriptions_.size();
    vertex_input_state.pVertexBindingDescriptions = vertex_binding->binding_descriptions_.data();
    vertex_input_state.vertexAttributeDescriptionCount = vertex_binding->attribute_descriptions_.size();
    vertex_input_state.pVertexAttributeDescriptions = vertex_binding->attribute_descriptions_.data();

    VkPipelineShaderStageCreateInfo shaderStages[2];
    shaderStages[0] = VkPipelineShaderStageCreateInfo{};
    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].pName = vertex_shader->get_entry_point();
    shaderStages[0].module = pipeline_key_cache_.shaders[0];

    shaderStages[1] = VkPipelineShaderStageCreateInfo{};
    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].pName = vertex_shader->get_entry_point();
    shaderStages[0].module = pipeline_key_cache_.shaders[1];

    VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.layout = pipeline_key_cache_.pipeline_layout;
    pipelineCreateInfo.renderPass = pipeline_key_cache_.render_pass;
    pipelineCreateInfo.subpass = 0;
    pipelineCreateInfo.stageCount = 2;
    pipelineCreateInfo.pStages = shaderStages;
    pipelineCreateInfo.pVertexInputState = &vertex_input_state;

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state = {};
    input_assembly_state.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly_state.topology = (VkPrimitiveTopology)pipeline_key_cache_.topology;
    pipelineCreateInfo.pInputAssemblyState = &input_assembly_state;
    VulkanPipeline pipeline_entry;
    VkResult error = vkCreateGraphicsPipelines(device->logicalDevice(), VK_NULL_HANDLE, 1, &pipelineCreateInfo,
                                               nullptr, &pipeline_entry.pipeline_);
    vulkan_pipelines_.insert(std::make_pair(pipeline_key_cache_, pipeline_entry));
    
    return std::make_tuple(pipeline_key_cache_.pipeline_layout, pipeline_entry);
}

void VulkanPipelineManager::clear(VulkanDevice *device) {
    for (auto &iter : vulkan_pipelines_) {
        vkDestroyPipeline(device->logicalDevice(), iter.second.pipeline_, nullptr);
    }
    vulkan_pipelines_.clear();
}

VkPipelineLayout VulkanPipelineManager::get_pipeline_layout(VulkanDevice *device, VkDescriptorSetLayout *descriptset_layouts, uint8_t descriptset_layouts_count) {
    assert(descriptset_layouts_count != 0);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = descriptset_layouts_count;
    pipelineLayoutInfo.pSetLayouts = descriptset_layouts;

    PipelineLayoutKey pipeline_layout_key;
    pipeline_layout_key.descriptor_set_count = descriptset_layouts_count;
    for (uint8_t i = 0; i < descriptset_layouts_count; ++i) {
        pipeline_layout_key.descriptor_set_layouts[i] = descriptset_layouts[i];
    }

    auto it = pipeline_layouts_.find(pipeline_layout_key);
    if (it != pipeline_layouts_.end()) {
        return it->second;
    }

    VkPipelineLayout pipeline_layout;
    vkCreatePipelineLayout(device->logicalDevice(), &pipelineLayoutInfo, nullptr, &pipeline_layout);

    pipeline_layouts_.insert(std::make_pair(pipeline_layout_key, pipeline_layout));
    return pipeline_layout;
}

}// namespace ocarina


