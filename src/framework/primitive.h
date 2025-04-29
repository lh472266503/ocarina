//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "core/concepts.h"
#include "core/thread_pool.h"
#include "rhi/params.h"
#include "rhi/graphics_descriptions.h"
#include "rhi/pipeline_state.h"

namespace ocarina {
class VertexBuffer;
class IndexBuffer;
template <class T>
class Shader;

class Primitive {
public:
    Primitive() = default;
    ~Primitive();

    //Primitive(Primitive &&right);
    //Primitive &operator=(Primitive &&right);
    
    void set_pipeline_state(const PipelineState &pipeline_state) { pipeline_state_ = pipeline_state; }
    const PipelineState &get_pipeline_state() const { return pipeline_state_; }

    using GeometryDataSetup = ocarina::function<void(Primitive&)>;

    void set_geometry_data_setup(GeometryDataSetup setup) { geometry_data_setup_ = setup; }
    void set_vertex_buffer(VertexBuffer *vertex_buffer);
    void set_index_buffer(IndexBuffer *index_buffer);
    void set_vertex_shader(handle_ty vertex_shader);
    void set_pixel_shader(handle_ty pixel_shader);
    void set_blend_state(const BlendState &blend_state) { pipeline_state_.blend_state = blend_state; }
    void set_raster_state(const RasterState &raster_state) { pipeline_state_.raster_state = raster_state; }
    void set_depth_stencil_state(const DepthStencilState &depth_stencil_state) { pipeline_state_.depth_stencil_state = depth_stencil_state; }
    void set_primitive_type(PrimitiveType primitive_type) { pipeline_state_.primitive_type = primitive_type; }

    handle_ty get_vertex_shader() const { return vertex_shader_; }   
    handle_ty get_pixel_shader() const { return pixel_shader_; }
    VertexBuffer *get_vertex_buffer() const { return vertex_buffer_; }
    IndexBuffer *get_index_buffer() const { return index_buffer_; }

    void set_per_object_descriptors();

    void set_position(const float3 &position) {
        position_ = position;
        transform_dirty_ = true;
    }

    const float3 &get_position() const { return position_; }

    const float4x4& get_world_matrix() {
        if (transform_dirty_) {
            //world_matrix_ = math::translate(position_);
            transform_dirty_ = false;
        }
        return world_matrix_;
    }

private:
    VertexBuffer* vertex_buffer_;
    IndexBuffer* index_buffer_;
    handle_ty vertex_shader_;
    handle_ty pixel_shader_;
    PipelineState pipeline_state_;
    //std::unique_ptr<VertexBuffer> vertex_buffer_;
    GeometryDataSetup geometry_data_setup_;

    float4x4 world_matrix_;
    float3 position_;
    bool transform_dirty_ = true;
};

}// namespace ocarina