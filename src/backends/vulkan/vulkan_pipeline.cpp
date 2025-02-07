
#include "vulkan_device.h"
#include "vulkan_pipeline.h"
#include "util.h"
#include "vulkan_shader.h"

namespace ocarina {

void VulkanPipelineManager::bind_shader(const VulkanShader &shader, int stage) {
    pipeline_key_cache_.shaders[stage] = shader.shader_module();
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

VulkanPipeline VulkanPipelineManager::get_or_create_pipeline(const PipelineState &pipeline_state, VulkanDevice *device) {
    

    for (int i = 0; i < PipelineKey::MAX_SHADER_STAGE; ++i) {
        bind_shader(*(pipeline_state.shaders[i]), i);
    }
    bind_blend_state(pipeline_state.blend_state);
    bind_depth_stencil_state(pipeline_state.depth_stencil_state);
    bind_raster_state(pipeline_state.raster_state);
    bind_vertex_attributes(pipeline_state.vertex_info.attribute_descriptions.data(), pipeline_state.vertex_info.binding_descriptions.data(), 
        pipeline_state.vertex_info.attribute_descriptions.size(), pipeline_state.vertex_info.binding_descriptions.size());
    bind_topology(pipeline_state.primitive_type);

    auto it = vulkan_pipelines_.find(pipeline_key_cache_);
    if (it != vulkan_pipelines_.end()) {
        return it->second;
    }

    VkPipelineVertexInputStateCreateInfo vertex_input_state = {};
    vertex_input_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_state.vertexBindingDescriptionCount = pipeline_state.vertex_info.binding_descriptions.size();
    vertex_input_state.pVertexBindingDescriptions = pipeline_state.vertex_info.binding_descriptions.data();
    vertex_input_state.vertexAttributeDescriptionCount = pipeline_state.vertex_info.attribute_descriptions.size();
    vertex_input_state.pVertexAttributeDescriptions = pipeline_state.vertex_info.attribute_descriptions.data();

    VkPipelineShaderStageCreateInfo shaderStages[2];
    shaderStages[0] = VkPipelineShaderStageCreateInfo{};
    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].pName = pipeline_state.shaders[0]->get_entry_point();
    shaderStages[0].module = pipeline_key_cache_.shaders[0];

    shaderStages[1] = VkPipelineShaderStageCreateInfo{};
    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].pName = pipeline_state.shaders[1]->get_entry_point();
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
    
    return pipeline_entry;
}

void VulkanPipelineManager::clear(VulkanDevice *device) {
    for (auto &iter : vulkan_pipelines_) {
        vkDestroyPipeline(device->logicalDevice(), iter.second.pipeline_, nullptr);
    }
    vulkan_pipelines_.clear();
}

}// namespace ocarina


