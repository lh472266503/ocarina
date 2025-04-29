//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "graphics_descriptions.h"

namespace ocarina {
template<typename T>
class Shader;
class VertexBuffer;

struct RasterState {
    CullingMode cull_mode : 2;
    bool front_face : 1;
    bool depth_bias : 1;
    bool depth_clamp : 1;
    int padding : 27;
};

struct BlendState {
    BlendFunction srccolorblend_factor : 5;
    BlendFunction dstcolorblend_factor : 5;
    BlendFunction srcalphablend_factor : 5;
    BlendFunction dstalphablend_factor : 5;
    BlendOperator colorBlendOp : 4;
    BlendOperator alphaBlendOp : 4;
    int padding : 4;
};

struct DepthStencilState {
    bool depth_test_enable : 1;
    bool depth_write_enable : 1;
    SamplerCompareFunc depth_compare_op : 4;
    bool depth_bounds_test_enable : 1;
    bool stencil_test_enable : 1;
    int padding : 25;
};

struct PipelineState {
    static constexpr uint16_t MAX_SHADER_STAGE = 3;
    handle_ty shaders[MAX_SHADER_STAGE];
    //std::vector<VertexAttribute> vertex_attributes;
    //std::vector<VertexBinding> vertex_bindings;
    //VulkanVertexInfo vertex_info;
    //Handle<HwVertexBufferInfo> vertexBufferInfo;           
    //PipelineLayout pipelineLayout;            
    VertexBuffer *vertex_buffer = nullptr;             
    RasterState raster_state;//  4
    BlendState blend_state;
    DepthStencilState depth_stencil_state;                 
    PrimitiveType primitive_type = PrimitiveType::TRIANGLES;
};

}// namespace ocarina