//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "primitive.h"
#include "rhi/vertex_buffer.h"
#include "rhi/index_buffer.h"

namespace ocarina {


Primitive::~Primitive() {
    if (vertex_buffer_) {
        ocarina::deallocate<VertexBuffer>(vertex_buffer_);
    }

    if (index_buffer_) {
        ocarina::deallocate<IndexBuffer>(index_buffer_);
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

}// namespace ocarina