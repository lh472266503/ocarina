//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "vertex_buffer.h"
#include "device.h"

namespace ocarina {
VertexBuffer::~VertexBuffer() {
    release_vertex_buffer(device_, this);
}

VertexBuffer *VertexBuffer::create_vertex_buffer(Device::Impl *device) {
    VertexBuffer *buffer = device->create_vertex_buffer();
    buffer->device_ = device;
    //buffer->buffer_handle_ =  device->create_vertex_buffer(initial_data, bytes);
    return buffer;
}

void VertexBuffer::release_vertex_buffer(Device::Impl *device, VertexBuffer *vertex_buffer) {
    for (auto& stream : vertex_buffer->vertex_streams_) {
        if (stream.data) {
            delete[] stream.data;
            stream.data = nullptr;
        }
        if (stream.buffer_handle != InvalidUI64) {
            device->destroy_buffer(stream.buffer_handle);
            stream.buffer_handle = InvalidUI64;
        }
    }
}

void VertexBuffer::add_vertex_stream(VertexAttributeType::Enum type, uint32_t count, uint32_t stride, const void *data) {
    if (vertex_streams_[(uint8_t)type].data) {
        delete[] vertex_streams_[(uint8_t)type].data;
    }

    if (data != nullptr)
    {
        vertex_streams_[(uint8_t)type].data = new uint8_t[count * stride];
        memcpy(vertex_streams_[(uint8_t)type].data, data, count * stride);
    }

    vertex_streams_[(uint8_t)type].count = count;
    vertex_streams_[(uint8_t)type].stride = stride;
    vertex_streams_[(uint8_t)type].offset = 0;
}

}// namespace ocarina