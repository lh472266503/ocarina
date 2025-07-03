//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "primitive.h"
#include "rhi/vertex_buffer.h"
#include "rhi/index_buffer.h"
#include "rhi/device.h"
#include "rhi/descriptor_set.h"

namespace ocarina {


Primitive::~Primitive() {
    if (vertex_buffer_) {
        ocarina::delete_with_allocator<VertexBuffer>(vertex_buffer_);
    }

    if (index_buffer_) {
        ocarina::delete_with_allocator<IndexBuffer>(index_buffer_);
    }
}

void Primitive::set_geometry_data_setup(GeometryDataSetup setup) {
    geometry_data_setup_ = setup;
    if (geometry_data_setup_) {
        geometry_data_setup_(*this);
    }
}

void Primitive::set_vertex_buffer(VertexBuffer *vertex_buffer) {
    vertex_buffer_ = vertex_buffer;
    pipeline_state_.vertex_buffer = vertex_buffer;
}

void Primitive::set_index_buffer(IndexBuffer *index_buffer) {
    index_buffer_ = index_buffer;
}

void Primitive::set_vertex_shader(handle_ty vertex_shader) {
    vertex_shader_ = vertex_shader;
    pipeline_state_.shaders[0] = vertex_shader;
}

void Primitive::set_pixel_shader(handle_ty pixel_shader) {
    pixel_shader_ = pixel_shader;
    pipeline_state_.shaders[1] = pixel_shader;
}

DrawCallItem Primitive::get_draw_call_item(Device* device)
{
    DrawCallItem item;
    item.pipeline_state = &pipeline_state_;
    item.index_buffer = index_buffer_;

    if (descriptor_set_layout_ == nullptr)
    {
        void **shaders = reinterpret_cast<void**>(pipeline_state_.shaders);
        descriptor_set_layout_ = device->create_descriptor_set_layout(shaders, 2);
    }

    if (descriptor_set_ == nullptr) {
        //descriptor_set_ = std::make_unique<DescriptorSet>(descriptor_set_layout_->allocate_descriptor_set());
    }
    return item;
}

}// namespace ocarina