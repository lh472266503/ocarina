#pragma once
#include "core/header.h"
#include "core/concepts.h"
#include "core/stl.h"
#include "core/util.h"
#include "rhi/graphics_descriptions.h"
#include <vulkan/vulkan.h>
#include <functional>
#include "vulkan_buffer.h"
namespace ocarina {

class VulkanShader;
class VulkanDevice;

struct RasterState {
    VkCullModeFlags cull_mode : 2;
    VkFrontFace front_face : 2;
    VkBool32 depth_bias : 1;
    VkBool32 depth_clamp : 1;
    int padding : 26;
};

struct BlendState {
    VkBlendFactor srccolorblend_factor : 5;
    VkBlendFactor dstcolorblend_factor : 5;
    VkBlendFactor srcalphablend_factor : 5;
    VkBlendFactor dstalphablend_factor : 5;
    BlendOperator colorBlendOp : 4;
    BlendOperator alphaBlendOp : 4;
    int padding : 4;
};

struct DepthStencilState {
    VkBool32 depth_test_enable : 1;
    VkBool32 depth_write_enable : 1;
    SamplerCompareFunc depth_compare_op : 4;   
    VkBool32 depth_bounds_test_enable : 1;
    VkBool32 stencil_test_enable : 1;
    int padding : 25;
};

struct VertexInputAttributeDescription {
    VertexInputAttributeDescription &operator=(const VkVertexInputAttributeDescription &that) {
        location = that.location;
        binding = that.binding;
        format = that.format;
        offset = that.offset;
        return *this;
    }
    operator VkVertexInputAttributeDescription() const {
        return {location, binding, VkFormat(format), offset};
    }

    bool operator==(const VertexInputAttributeDescription& other) const
    {
        return location == other.location && binding == other.binding && format == other.format && offset == other.format;
    }
    uint8_t location = 0;
    uint8_t binding = 0;
    uint16_t format = 0;
    uint32_t offset = 0;
};

// Equivalent to VkVertexInputBindingDescription but not as big.
struct VertexInputBindingDescription {
    VertexInputBindingDescription &operator=(const VkVertexInputBindingDescription &that) {
        binding = that.binding;
        stride = that.stride;
        inputRate = that.inputRate;
        return *this;
    }
    operator VkVertexInputBindingDescription() const {
        return {binding, stride, (VkVertexInputRate)inputRate};
    }

    bool operator==(const VkVertexInputBindingDescription& other) const
    {
        return binding == other.binding && inputRate == other.inputRate && stride == other.stride;
    }
    uint16_t binding = 0;
    uint16_t inputRate = 0;
    uint32_t stride = 0;
};

struct PipelineKey
{
    static constexpr uint16_t MAX_VERTEX_ATTRIBUTES = 16;
    static constexpr uint16_t MAX_SHADER_STAGE = 2;
    VkShaderModule shaders[MAX_SHADER_STAGE] = {VK_NULL_HANDLE};
    VkRenderPass render_pass;
    RasterState raster_state;
    BlendState blend_state;
    DepthStencilState depth_stencil_state;
    VkPipelineLayout pipeline_layout;
    VkPrimitiveTopology topology;
    VertexInputAttributeDescription vertex_input_attributes[MAX_VERTEX_ATTRIBUTES];
    VertexInputBindingDescription vertex_input_binding[MAX_VERTEX_ATTRIBUTES];

    //we don't need to consider the vertex attribute here, because vertex attributes are assiociate to shader module.
    bool operator==(const PipelineKey &other) const {
        return *(int *)&raster_state == *(int *)&other.raster_state &&
               render_pass == other.render_pass &&
               shaders[0] == other.shaders[0] &&
               shaders[1] == other.shaders[1] &&
               *(int *)&blend_state == *(int *)&other.blend_state &&
               *(int *)&depth_stencil_state == *(int *)&other.depth_stencil_state && 
            topology == other.topology &&
            pipeline_layout == other.pipeline_layout;
    }
};

struct PipelineState {
    static constexpr uint16_t MAX_SHADER_STAGE = 3;
    VulkanShader *shaders[MAX_SHADER_STAGE];   
    VulkanVertexInfo vertex_info;
    //Handle<HwVertexBufferInfo> vertexBufferInfo;           //  4
    //PipelineLayout pipelineLayout;                         // 16
    RasterState raster_state;                               //  4
    BlendState blend_state;
    DepthStencilState depth_stencil_state;                             // 12
    PrimitiveType primitive_type = PrimitiveType::TRIANGLES;//  1
    //uint8_t padding[3] = {};                               //  3
};


struct HashPipelineKeyFunction
{
    uint64_t operator()(const PipelineKey &pipeline_key) const {
        std::size_t res = 0;
        hash_combine(res, *((uint64_t*)&pipeline_key.raster_state));
        hash_combine(res, *((uint64_t *)&pipeline_key.blend_state));
        hash_combine(res, *((uint64_t *)&pipeline_key.depth_stencil_state));
        hash_combine(res, pipeline_key.render_pass);
        hash_combine(res, (uint64_t)pipeline_key.shaders[0] << 32 | (uint64_t)pipeline_key.shaders[1]);
        hash_combine(res, pipeline_key.pipeline_layout);
        hash_combine(res, pipeline_key.topology);
        return res;
    }

    
};


struct VulkanPipeline {
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

class VulkanPipelineManager : public concepts::Noncopyable
{
public:
    void bind_shader(const VulkanShader &shader, int stage);
    void bind_raster_state(const RasterState &raster_state);
    void bind_blend_state(const BlendState &blend_state);
    void bind_depth_stencil_state(const DepthStencilState &depth_stencil_state);
    void bind_pipeline_layout(VkPipelineLayout pipeline_layout);
    void bind_vertex_attributes(VkVertexInputAttributeDescription const *attributes,
                                VkVertexInputBindingDescription const *binds, uint8_t attr_count, uint8_t bind_desc_count);
    void bind_topology(PrimitiveType primitive_type);
    VulkanPipeline get_or_create_pipeline(const PipelineState& pipeline_state, VulkanDevice* device);
    void clear(VulkanDevice *device);

private:
    std::unordered_map<PipelineKey, VulkanPipeline, HashPipelineKeyFunction> vulkan_pipelines_;
    PipelineKey pipeline_key_cache_;
};
}// namespace ocarina